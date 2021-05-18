import jax.numpy as jnp
from jax import jit, eval_shape
from functools import partial
from typing import Callable, Tuple, Union
from jwave import geometry
from jwave.geometry import Staggered

# TODO: Use map / fori / scan instead of for loops
# TODO: speed tests


def derivative_init(
    sample_input: jnp.ndarray,
    grid: geometry.kGrid,
    axis: int = -1,
    staggered: Staggered = Staggered.NONE,
    kspace_op: bool = False,
) -> Tuple[Callable, geometry.kGrid]:
    """[summary]

    Args:
        grid (geometry.kGrid): Reference grid
        x (jnp.ndarray): Sample input signal, for shape and dtype evaluation
        axis (int, optional): Axis on which the derivative is performed. Defaults to
            the last axis.
        degree (float, optional): Degree of the derivative operator.
        staggered (Staggered, optional): Staggered flag.
        kspace_op (bool, optional): If `True`, uses a modified derivative corrected by
            the k-space operator.

    Returns:
        Returns the spectral derivative operator `D` as
            first output and the updated `grid` object as second output. The function is
            used as `dy = D(y, grid)`, where the input signal `y` must have the same
            shape and type as `x`
    """
    # Update and check the grid object

    if staggered != Staggered.NONE:
        grid = grid.add_staggered_grid()
    if kspace_op:
        if grid.k_with_kspaceop is None:
            raise AssertionError(
                "You asked for a derivative with kspace operator but the kGrid object "
                + "has an empty k_with_kspaceop field. Did you call grid.apply_kspace_operator first?"
            )

    # Building the derivative operator
    if kspace_op:
        derivative_operator = _derivative_with_k_op(sample_input, staggered, axis)
    else:
        derivative_operator = _plain_derivative(sample_input, staggered, axis)

    return derivative_operator, grid


def _derivative_with_k_op(
    sample_x: jnp.ndarray, staggered: Staggered, axis: Union[int, Tuple[int]]
) -> Callable:
    def deriv_fun(x: jnp.ndarray, grid: geometry.kGrid) -> jnp.ndarray:
        # Selecting operator
        K = 1j * grid.k_with_kspaceop[staggered]

        n_dims = len(K) + 1
        domain_axes = list(range(-1, -n_dims, -1))

        # Choose FFT and filter according to signal domain (real or complex)
        if "complex" in str(x.dtype):
            return jnp.fft.ifftn(
                K[axis] * jnp.fft.fftn(x, axes=domain_axes), axes=domain_axes
            )
        else:
            Fx = eval_shape(
                lambda signal: jnp.fft.rfftn(signal, axes=domain_axes), sample_x
            )
            K = K[axis, : Fx.shape[0]]
            return jnp.fft.irfftn(
                K * jnp.fft.rfftn(x, axes=domain_axes), axes=domain_axes
            )

    return deriv_fun


def _plain_derivative(
    sample_x: jnp.ndarray, staggered: Staggered, axis: int
) -> Callable:
    def deriv_fun(x: jnp.ndarray, grid: geometry.kGrid, degree: float) -> jnp.ndarray:
        # Select operator
        k = 1j * grid.k_vec[staggered][axis]
        k = k ** degree

        if "complex" in str(sample_x.dtype):
            ffts = [jnp.fft.fft, jnp.fft.ifft]
        else:
            ffts = [jnp.fft.rfft, jnp.fft.irfft]
            x0 = jnp.moveaxis(sample_x, axis, -1)
            Fx = eval_shape(lambda x: ffts[0](x, axis=-1), x0)
            k = k[: Fx.shape[-1]]

        # Work on last axis for elementwise product broadcasting
        # TODO: This may be very inefficient: https://github.com/google/jax/issues/2972
        x = jnp.moveaxis(x, axis, -1)

        # Use real or complex fft
        Fx = ffts[0](x, axis=-1)
        dx = ffts[1](k * Fx, axis=-1, n=x.shape[-1])

        # Back to original axis
        dx = jnp.moveaxis(dx, -1, axis)
        return dx

    return deriv_fun
