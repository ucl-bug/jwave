from jax import numpy as jnp
from jax import lax
from jax._src.numpy.lax_numpy import (
    where,
    _sinc_maclaurin,
    pi,
    _check_arraylike,
    _promote_dtypes_inexact,
)


def safe_sinc(x: jnp.ndarray) -> jnp.ndarray:
    _check_arraylike("sinc", x)
    (x,) = _promote_dtypes_inexact(x)
    eq_zero = lax.eq(x, lax._const(x, 0))
    pi_x = lax.mul(lax._const(x, pi), x)
    safe_pi_x = where(eq_zero, lax._const(x, 1), pi_x)
    return where(
        eq_zero, _sinc_maclaurin(0, pi_x), lax.div(lax.sin(safe_pi_x), safe_pi_x)
    )

def join_dicts(dict1, dict2):
    for k, v in dict2.items():
        if k in dict1:
            continue
        else:
            dict1[k] = v
    return dict1