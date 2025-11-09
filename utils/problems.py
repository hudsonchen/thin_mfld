from dataclasses import dataclass
from typing import Callable, Optional
import jax.numpy as jnp
import jax
from typing import NamedTuple
from jaxtyping import Array 


@dataclass(frozen=True)
class Problem:
    # R1' : R -> R
    particle_d: int
    input_d: int
    output_d: int
    R1_prime: Callable[[Array], Array]
    # q1 : R^d -> R
    q1: Callable[[Array], Array]
    # Optional q2 : R^d × R^d -> R, used if gradx_q2 is None
    q2: Callable[[Array, Array], Array] = None
    # Optional ∇_x q2 : R^d × R^d -> R^d (preferred for performance)
    gradx_q2: Callable[[Array, Array], Array] = None
    data: Optional[Array] = None


class NNParams(NamedTuple):
    W1: jnp.ndarray  # (hidden, d)
    b1: jnp.ndarray  # (hidden,)
    W2: jnp.ndarray  # (hidden,)
    b2: jnp.ndarray  # () scalar

def init_nn_params(key, d: int, hidden: int = 64) -> NNParams:
    k1, k2, k3 = jax.random.split(key, 3)
    W1 = jax.random.normal(k1, (hidden, d)) / jnp.sqrt(d)
    b1 = jnp.zeros((hidden,))
    W2 = jax.random.normal(k2, (hidden,)) / jnp.sqrt(hidden)
    b2 = jnp.array(0.0)
    return NNParams(W1, b1, W2, b2)

def q1_nn_apply(params: NNParams, x: jnp.ndarray) -> jnp.ndarray:
    # x: (d,), returns scalar
    h = jnp.tanh(params.W1 @ x + params.b1)
    return jnp.dot(params.W2, h) + params.b2