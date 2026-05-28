# ----------------------- Config -----------------------
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

@dataclass
class CFG:
    N: int = 2048
    steps: int = 5000
    step_size: float = 1e-3
    sigma: float = 1.0
    zeta : float = 1e-2
    seed: int = 0
    kernel: str = "sobolev"  # "sobolev" or "gaussian"
    g : int = 0  # parameter for KT thinning
    bandwidth: float = 1.0  # for Gaussian kernel
    # Return full trajectory if True (memory heavy): (steps+1, N, d)
    return_path: bool = False
    kt_function: str = "compresspp_kt"
    skip_swap: bool = True

