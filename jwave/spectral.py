import jax.numpy as jnp
from jax import jit
from functools import partial
from jwave import geometry

# TODO: Use map / fori / scan instead of for loops
# TODO: speed tests


def diag_nabla_with_k_op(
    x: jnp.ndarray, grid: geometry.kGrid, staggered: int
):
    r = []
    for i in range(x.shape[0]):
        r.append(derivative_with_k_op(x[i], grid, staggered, domain, i))
    return jnp.stack(r, axis=0)


def derivative(
    x: jnp.ndarray,
    grid: geometry.kGrid,
    staggered: int,
    axis: int,
    kspace_op=False,
    degree: int = 1,
) -> jnp.ndarray:
    """Evaluates the spatial derivative of an n-dimensional
    array using spectral methods.

    Args:
        x (jnp.ndarray): [description]
        grid (geometry.kGrid): [description]
        staggered (int): [description]
        axis (int): [description]
        kspace_op (Bool, optional): [description]. Defaults to False.
        degree (int, optional): [description]. Defaults to 1.

    Returns:
        jnp.ndarray: [description]
    """
    if kspace_op:
        # TODO: warn user about the fact that the degree and parameter
        # is ignored
        dx = derivative_with_k_op(x, grid, staggered, axis)
    else:
        dx = plain_derivative(x, grid, staggered, axis, degree)
    return dx


# --------------------------------------------
def derivative_with_k_op(x, grid, staggered, axis):
    # Selecting operator
    if staggered == 0:
        K = 1j * grid.k_with_kspaceop["plain"]
    elif staggered == -1:
        K = 1j * grid.k_with_kspaceop["backward"]
    elif staggered == 1:
        K = 1j * grid.k_with_kspaceop["forward"]

    n_dims = len(K) + 1
    domain_axes = list(range(-1, -n_dims, -1))

    # Make fft
    Fx = jnp.fft.fftn(x, axes=domain_axes)

    # batched filtering
    kx = jnp.fft.ifftn(K[axis] * Fx, axes=domain_axes)
    return jnp.asarray(kx, x.dtype)

"""
# VJP rule ready, but crashes with linear_transpose() required
# by GMRES
def _derivative_with_k_op_fwd(x, k, domain_axes, domain):
    dx = _derivative_with_k_op(x, k, domain_axes, domain)
    return dx, (k,)


def _derivative_with_k_op_bwd(domain_axes, domain, res, g):
    k = res[0]
    # Only first order derivative can be computed, hence
    # the minus sign
    grad_x = -_derivative_with_k_op(g, k, domain_axes, domain)
    return (grad_x, jnp.zeros_like(k))


_derivative_with_k_op.defvjp(
    _derivative_with_k_op_fwd, _derivative_with_k_op_bwd
)
"""
# --------------------------------------------


def plain_derivative(x, grid, staggered, axis, degree):
    # Work on last axis for elementwise product broadcasting
    x = jnp.moveaxis(x, axis, -1)

    # Spectral derivative
    if staggered == 0:
        k = 1j * grid.k_vec[axis]
    elif staggered == -1:
        k = 1j * grid.k_staggered["backward"][axis]
    elif staggered == 1:
        k = 1j * grid.k_staggered["forward"][axis]

    dx = _derivative_algorithm_last_axis(x, k, degree)

    # Back to original axis
    kx = jnp.moveaxis(dx, -1, axis)
    return kx


# @partial(jax.custom_vjp, nondiff_argnums=(2,3))
def _derivative_algorithm_last_axis(x, k, degree):
    k = k ** degree
    Fx = jnp.fft.fft(x, axis=-1)
    kx = jnp.fft.ifft(k * Fx, axis=-1)
    return jnp.asarray(kx, x.dtype)


"""
# VJP rule ready, but crashes with linear_transpose() required
# by GMRES
def _derivative_algorithm_last_axis_fwd(x, k, degree, domain):
    k = k ** degree
    if domain == "real":
        Fx = jnp.fft.rfft(x, axis=-1)
        k_short = k[: Fx.shape[-1]]
        dx = jnp.fft.irfft(k_short * Fx, axis=-1)
    else:
        dx = jnp.fft.ifft(k * jnp.fft.fft(x, axis=-1), axis=-1)
    return dx, (k,)


def _derivative_algorithm_last_axis_bwd(degree, domain, res, g):
    k, = res
    sign = (-1.0) ** degree
    grad_x = sign * _derivative_algorithm_last_axis(g, k, 1, domain)
    grad_k = jnp.zeros_like(
        k
    )  # Cant differentiate the k vector TODO: remove constraint
    return (grad_x, grad_k)


_derivative_algorithm_last_axis.defvjp(
    _derivative_algorithm_last_axis_fwd, _derivative_algorithm_last_axis_bwd
)
"""

# --------------------------------------------


@partial(jit, static_argnums=(2, 3))
def gradient(x, grid, staggered, domain):
    grads = []
    for i in range(len(x.shape)):
        grads.append(derivative(x, i, grid, staggered, domain))
    return jnp.stack(grads)


@partial(jit, static_argnums=(2, 3))
def laplacian(x, grid, staggered, domain):
    L = jnp.zeros_like(x)
    for ax in range(len(x.shape)):
        L += derivative(
            derivative(x, ax, grid, staggered, domain), ax, grid, staggered, domain
        )
    return L


@partial(jit, static_argnums=(2, 3))
def divergence(x, grid, staggered, domain):
    div = jnp.zeros_like(x[0])
    for i in range(x.shape[0]):
        div += derivative(x[i], i, grid, staggered, domain)
    return div


@partial(jit, static_argnums=(2, 3))
def diag_nabla(x, grid, staggered, domain):
    r = []
    for i in range(x.shape[0]):
        r.append(derivative(x[i], grid, staggered, domain, i))
    return jnp.stack(r, axis=0)
