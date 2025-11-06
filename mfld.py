
# JAX mean-field Langevin dynamics as a reusable class.
# v_mu(x) = R1'( E[q1] ) ∇q1(x) + E[ ∇_x q2(x, X̃) ]
# dX_t = -v_mu(X_t) step_size + sqrt(2/β) dW_t

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional
import math
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap, grad, random, lax
from utils.configs import CFG
from utils.problems import Problem
from utils.kernel import compute_mmd2
from jaxtyping import Array 
from jax_tqdm import scan_tqdm
# from utils.kt import kt_compresspp
from tqdm import tqdm
from goodpoints.jax.compress import kt_compresspp
from goodpoints import compress
from goodpoints.jax.sliceable_points import SliceablePoints


def glorot_normal(key, fan_in, fan_out):
    std = jnp.sqrt(2.0 / (fan_in + fan_out))
    return std * jax.random.normal(key, (fan_in, fan_out))

def initialize(key, d_in, d_hidden, d_out):
    """PyTorch-like initialization for a 2-layer tanh MLP."""
    k1, k2 = random.split(key)
    k3, k4 = random.split(k2)
    W1 = glorot_normal(k1, d_in, d_hidden)
    b1 = jnp.zeros((d_hidden,))
    W2 = glorot_normal(k3, d_hidden, d_out)
    return W1, b1, W2



def uncentered_matern_32_kernel(points_x, points_y, l):
    X, Y = points_x.get("X"), points_y.get("X")  # (N_x, d), (N_y, d)
    # diff = X[:, None, :] - Y[None, :, :]         # (N_x, N_y, d)
    diff = X - Y
    dists = jnp.linalg.norm(diff, axis=-1)       # (N_x, N_y)
    sqrt3_r = jnp.sqrt(3.0) * dists / l
    return (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r)   # (N_x, N_y)


class MFLD:
    def __init__(self, thinning, cfg: CFG, problem: Problem):
        self.cfg = cfg
        self.problem = problem
        self.data = problem.data
        self.counter = 0
        self.seed = cfg.seed
        self.kernel_type = cfg.kernel
        if self.kernel_type == "sobolev":
            k_params = np.array([1.0, 2.0, 3.0]) 
        elif self.kernel_type == "gaussian":
            k_params = np.array([self.cfg.bandwidth])
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")


        # np.random.seed(self.seed)

        if thinning == 'kt':
            # This is the jax version of kt_compresspp, which is very slow.
            #
            kernel_fn = partial(uncentered_matern_32_kernel, l=float(1.0))
            rng = np.random.default_rng(self.seed)
            def thin_fn(x):
                points = SliceablePoints({"X": x})  
                coresets = kt_compresspp(kernel_fn, points, w=jnp.ones(self.cfg.N) / self.cfg.N, 
                             rng_gen=rng, inflate_size=int(self.cfg.N), g=0, delta=0.5)
                return x[coresets, :]
            # This is the cython version which is fast
            # def thin_fn(x):
            #     x_cpu = np.array(np.asarray(x))
            #     coresets = compress.compresspp_kt(x_cpu, kernel_type=self.kernel_type.encode("utf-8"), k_params=k_params, seed=self.seed, g=0)
            #     return jax.device_put(x_cpu[coresets, :])
            self.thin_fn = thin_fn
        elif thinning == 'random':
            def thin_fn(x):
                rng = np.random.default_rng(self.seed + self.counter)
                indices = rng.choice(self.cfg.N, size=int(jnp.sqrt(self.cfg.N)), replace=False)
                return x[indices, :]
            self.thin_fn = thin_fn
        elif thinning == 'false':
            self.thin_fn = lambda x: x
        else:
            raise ValueError(f"Unknown thinning method: {thinning}")

        self._vm_q1 = vmap(vmap(self.problem.q1, in_axes=(None, 0)), in_axes=(0, None))  # (N,d) -> (N,)
        self._vm_grad_q1 = vmap(vmap(grad(self.problem.q1, argnums=1), in_axes=(None, 0)), in_axes=(0, None))      # (N,d) -> (N,d)

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
        # s = self._vm_q1(self.data["Z"][self.counter, ...], thinned_x) - self.data["y"][self.counter, ...][:, None]  # (n, N)
        s = self._vm_q1(self.data["Z"][self.counter, ...], thinned_x).sum(1) * (x.shape[0] / thinned_x.shape[0]) - self.data["y"][self.counter, ...]  # (n, )
        coeff = self.problem.R1_prime(s)   
        # term1_coeff = jnp.mean(coeff, axis=1)    # (n, )         
        term1_vector = self._vm_grad_q1(self.data["Z"][self.counter, ...], x)       # (n, N, d)
        term1_mean = (coeff[:, None, None] * term1_vector).mean(0)  # (N,d)

        # Second term: mean over j of ∇_x q2(x_i, x_j)
        gx = self._all_pairs_gx(x)                   # (N,N,d)
        term2 = jnp.mean(gx, axis=1)                 # (N,d)
        # Regularization 
        term3 = self.cfg.zeta * x
        return term1_mean + term2 + term3

    
    # @partial(jit, static_argnums=0)
    def _step(self, carry, _):
        x, key = carry
        thinned_x = self.thin_fn(x)

        # Debug code compare MMD between x and thinned_x 
        mmd2 = compute_mmd2(x, thinned_x, bandwidth=1.0)
        ###########################################

        v = self.vector_field(x, thinned_x)
        noise_scale = jnp.sqrt(2.0 * self.cfg.sigma * self.cfg.step_size)
        key, sub = random.split(key)
        noise = noise_scale * random.normal(sub, shape=x.shape)
        x_next = x - self.cfg.step_size * v + noise
        self.counter = (self.counter + 1) % self.data["batch_size"]
        return (x_next, mmd2, key), x_next

    def simulate(self, x0: Optional[Array] = None) -> Array:
        key = random.PRNGKey(self.cfg.seed)
        if x0 is None:
            key, sub = random.split(key)
            # x0 = 0.5 * random.normal(sub, (self.cfg.N, self.problem.particle_d)) * 0.1
            W1_0, b1_0, W2_0 = initialize(key, self.problem.data_d, d_hidden=self.cfg.N, d_out=1)
            x0 = jnp.concatenate([W1_0.T, b1_0[:, None], W2_0], axis=1)  # (N, d)

        x = x0
        path = []
        mmd_path = []
        for t in tqdm(range(self.cfg.steps)):
            key_, subkey = random.split(key)
            (x, mmd2, key) , _ = self._step((x, subkey), t)
            path.append(x)
            mmd_path.append(mmd2)

        path = jnp.stack(path, axis=0)          # (steps, N, d)
        mmd_path = jnp.stack(mmd_path, axis=0)  # (steps, )

        return path, mmd_path
