# JAX mean-field Langevin dynamics as a reusable class.
# v_mu(x) = R1'( E[q1] ) ∇q1(x) + E[ ∇_x q2(x, X̃) ]
# dX_t = -v_mu(X_t) step_size + sqrt(2/β) dW_t

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, random, lax
from utils.configs import CFG
from utils.problems import Problem
from jaxtyping import Array 
from jax_tqdm import scan_tqdm


# ----------------------- Simulator class -----------------------
class MFLD:
    def __init__(self, cfg: CFG, problem: Problem):
        self.cfg = cfg
        self.problem = problem
        self.data = problem.data

        # Build vmapped helpers once (static w.r.t. self)
        self._vm_q1 = jax.vmap(jax.vmap(self.problem.q1, in_axes=(None, 0)), in_axes=(0, None))  # (N,d) -> (N,)
        self._vm_grad_q1 = jax.vmap(jax.vmap(grad(self.problem.q1, argnums=1), in_axes=(None, 0)), in_axes=(0, None))      # (N,d) -> (N,d)

        if self.problem.q2 is None and self.problem.gradx_q2 is None:
            gx = lambda x, y: 0 * x
            self._pair_gx = lambda xi, X: vmap(lambda yj: gx(xi, yj))(X)  # (N,d) -> (N,d)
        else:
            self._pair_gx = lambda xi, X: vmap(lambda yj: self.problem.gradx_q2(xi, yj))(X)  # (N,d) -> (N,d)

        # Vectorized over all (i,j) pairs: returns (N,N,d)
        self._all_pairs_gx = lambda X: vmap(lambda xi: self._pair_gx(xi, X))(X)

    # Treat `self` as static for JIT so the callables are constants.
    @partial(jit, static_argnums=0)
    def vector_field(self, x: Array) -> Array:
        # First term: R1'(E[q1]) * ∇q1(x)
        s = self._vm_q1(self.data["Z"], x) - self.data["y"][:, None]  # (n, N)
        coeff = self.problem.R1_prime(s)            
        term1 = coeff[:, :, None] * self._vm_grad_q1(self.data["Z"], x)       # (n, N, d)
        term1_mean = jnp.mean(term1, axis=0)               # (N,d)
        # Second term: mean over j of ∇_x q2(x_i, x_j)
        gx = self._all_pairs_gx(x)                   # (N,N,d)
        term2 = jnp.mean(gx, axis=1)                 # (N,d)
        return term1_mean + term2

    
    @partial(jit, static_argnums=0)
    def _step(self, carry, _):
        x, key = carry
        v = self.vector_field(x)
        noise_scale = jnp.sqrt(2.0 * self.cfg.sigma * self.cfg.step_size)
        key, sub = random.split(key)
        noise = noise_scale * random.normal(sub, shape=x.shape)
        x_next = x - self.cfg.step_size * v + noise
        return (x_next, key), x_next

    def simulate(self, x0: Optional[Array] = None) -> Array:
        key = random.PRNGKey(self.cfg.seed)
        if x0 is None:
            key, sub = random.split(key)
            x0 = 0.5 * random.normal(sub, (self.cfg.N, self.problem.d))

        scan_fn = scan_tqdm(self.cfg.steps)(self._step)
        (xT, _), path = lax.scan(scan_fn, (x0, key), jnp.arange(self.cfg.steps))

        if self.cfg.return_path:
            # Include initial state for convenience
            return jnp.concatenate([x0[None, ...], path], axis=0)  # (steps+1, N, d)
        return xT
