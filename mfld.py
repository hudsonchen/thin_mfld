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
from goodpoints.jax.compress import kt_compresspp


def initialize(key, d_in, d_hidden, d_out):
    """PyTorch-like initialization for a 2-layer tanh MLP."""
    k1, k2 = jax.random.split(key)
    k3, k4 = jax.random.split(k2)

    # Layer 1: Linear(d_in, d_hidden)
    # bound1 = jnp.sqrt(1.0 / d_in)
    bound1 = 1
    # W1 = jax.random.uniform(k1, (d_in, d_hidden), minval=-bound1, maxval=bound1)
    W1 = jax.random.normal(k1, (d_in, d_hidden)) * bound1
    # b1 = jnp.zeros((d_hidden,))
    b1 = jax.random.normal(k2, (d_hidden,)) * bound1

    # Layer 2: Linear(d_hidden, d_out)
    # bound2 = jnp.sqrt(1.0 / d_hidden)
    bound2 = 1
    # W2 = jax.random.uniform(k2, (d_hidden, d_out), minval=-bound2, maxval=bound2)
    W2 = jax.random.normal(k3, (d_hidden, d_out)) * bound2
    # b2 = jnp.zeros((d_out,))
    b2 = jax.random.normal(k4, (d_hidden, )) * bound2
    return W1, b1, W2, b2




# ----------------------- Simulator class -----------------------
class MFLD:
    def __init__(self, thinning, cfg: CFG, problem: Problem):
        self.cfg = cfg
        self.problem = problem
        self.data = problem.data
        self.counter = 0
        if thinning :
            self.thin_fn = lambda x: x[::self.cfg.thin_factor, ...]
        else:
            self.thin_fn = lambda x: x
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
    def vector_field(self, x: Array, thinned_x: Array) -> Array:
        # First term: R1'(E[q1]) * ∇q1(x)
        s = self._vm_q1(self.data["Z"][self.counter, ...], thinned_x) - self.data["y"][self.counter, ...][:, None]  # (n, N)
        coeff = self.problem.R1_prime(s)   
        term1_coeff = jnp.mean(coeff, axis=1)    # (n, )         
        term1_vector = self._vm_grad_q1(self.data["Z"][self.counter, ...], x)       # (n, N, d)
        term1_mean = (term1_coeff[:, None, None] * term1_vector).mean(0)  # (N,d)

        # Second term: mean over j of ∇_x q2(x_i, x_j)
        gx = self._all_pairs_gx(x)                   # (N,N,d)
        term2 = jnp.mean(gx, axis=1)                 # (N,d)
        # Regularization 
        term3 = self.cfg.zeta * x
        return term1_mean + term2 + term3

    
    @partial(jit, static_argnums=0)
    def _step(self, carry, _):
        x, key = carry
        thinned_x = self.thin_fn(x)
        v = self.vector_field(x, thinned_x)
        noise_scale = jnp.sqrt(2.0 * self.cfg.sigma * self.cfg.step_size)
        key, sub = random.split(key)
        noise = noise_scale * random.normal(sub, shape=x.shape)
        x_next = x - self.cfg.step_size * v + noise
        self.counter = (self.counter + 1) % self.data["batch_size"]
        return (x_next, key), x_next

    def simulate(self, x0: Optional[Array] = None) -> Array:
        key = random.PRNGKey(self.cfg.seed)
        if x0 is None:
            key, sub = random.split(key)
            # x0 = 0.5 * random.normal(sub, (self.cfg.N, self.problem.particle_d)) * 0.1
            W1_0, b1_0, W2_0, b2_0 = initialize(key, self.problem.data_d, d_hidden=self.cfg.N, d_out=1)
            x0 = jnp.concatenate([W1_0.T, b1_0[:, None], W2_0, b2_0[:, None]], axis=1)  # (N, d)

        scan_fn = scan_tqdm(self.cfg.steps)(self._step)
        (xT, _), path = lax.scan(scan_fn, (x0, key), jnp.arange(self.cfg.steps))

        if self.cfg.return_path:
            # Include initial state for convenience
            return jnp.concatenate([x0[None, ...], path], axis=0)  # (steps+1, N, d)
        return xT



