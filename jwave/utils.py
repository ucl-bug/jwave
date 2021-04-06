from jax import numpy as jnp
from jax import vmap, lax, tree_util
from jax import lax
from jax._src.numpy.lax_numpy import (
    where,
    _sinc_maclaurin,
    pi,
    _check_arraylike,
    _promote_dtypes_inexact,
)
from functools import partial
from matplotlib import pyplot as plt


def safe_sinc(x):
    _check_arraylike("sinc", x)
    (x,) = _promote_dtypes_inexact(x)
    eq_zero = lax.eq(x, lax._const(x, 0))
    pi_x = lax.mul(lax._const(x, pi), x)
    safe_pi_x = where(eq_zero, lax._const(x, 1), pi_x)
    return where(
        eq_zero, _sinc_maclaurin(0, pi_x), lax.div(lax.sin(safe_pi_x), safe_pi_x)
    )


def print_graph(node, indent=0):
    print("| " * indent, node)
    if len(node.parents) == 0:
        return
    else:
        for p in node.parents:
            print_graph(p, indent + 1)


def quickshow_helmholtz_results(medium, field):
    _, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(medium["sound_speed"])
    axes[0].set_title("Speed of sound")
    axes[1].imshow(field.real, vmin=-1, vmax=1, cmap="seismic")
    axes[1].set_title("Real wavefield")
    axes[2].imshow(jnp.abs(field), vmin=0, vmax=1, cmap="magma")
    axes[2].set_title("Wavefield intensity")


def assert_pytree_isclose(a, b, relative_precision=1e-4, abs_precision=1e-6):
    leaves_reference = tree_util.tree_leaves(a)
    leaves_output = tree_util.tree_leaves(b)
    assert len(leaves_output) != len(leaves_reference)

    is_close = map(
        lambda x: jnp.allclose(x[0], x[1], relative_precision, abs_precision),
        zip(leaves_reference, leaves_output),
    )
    assert all(is_close)
