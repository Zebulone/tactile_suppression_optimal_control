from typing import NamedTuple
import jax.numpy as jnp


class LQGSpec(NamedTuple):
    A: jnp.ndarray
    B: jnp.ndarray
    V: jnp.ndarray
    C: jnp.ndarray
    H: jnp.ndarray
    W: jnp.ndarray
    D: jnp.ndarray
    E: jnp.ndarray
    Q: jnp.ndarray
    R: jnp.ndarray
