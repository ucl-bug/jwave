from jax import numpy as jnp

from jwave.geometry import _circ_mask


def three_circles(N: tuple) -> jnp.ndarray:
    """
    Generate a 3-circle phantom.

    Args:
        N: The size of the phantom.

    Returns:
        jnp.ndarray: The phantom.
    """
    radius = sum(N) / float(len(N))
    mask1 = _circ_mask(N, radius * 0.05, (int(N[0] / 2 + N[0] / 8), int(N[1] / 2)))
    mask2 = _circ_mask(
        N, radius * 0.1, (int(N[0] / 2 - N[0] / 8), int(N[1] / 2 + N[1] / 6))
    )
    mask3 = _circ_mask(N, radius * 0.15, (int(N[0] / 2), int(N[1] / 2)))
    p0 = 5.0 * mask1 + 3.0 * mask2 + 4.0 * mask3
    return jnp.expand_dims(p0, -1)
