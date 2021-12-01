from jwave.geometry import _circ_mask
from jax import numpy as jnp


def three_circles(N: tuple) -> jnp.ndarray:
    """
    Generate a 3-circle phantom.

    Args:
        N: The size of the phantom.

    Returns:
        jnp.ndarray: The phantom.
    """
    mask1 = _circ_mask(N, 8, (50, 50))
    mask2 = _circ_mask(N, 5, (80, 60))
    mask3 = _circ_mask(N, 10, (64, 64))
    p0 = 5.0 * mask1 + 3.0 * mask2 + 4.0 * mask3
    return p0
