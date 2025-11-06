# ----------------------- Config -----------------------
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, random, lax
from jaxtyping import Array 

@dataclass
class CFG:
    N: int = 2048
    steps: int = 5000
    step_size: float = 1e-3
    sigma: float = 1.0
    zeta : float = 1e-2
    seed: int = 0
    kernel: str = "sobolev"  # "sobolev" or "gaussian"
    # Return full trajectory if True (memory heavy): (steps+1, N, d)
    return_path: bool = False

# ----------------------- Problem spec -----------------------
