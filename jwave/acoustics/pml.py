from typing import Callable

import jaxdf.operators as jops
from jax import numpy as jnp
from jaxdf.core import Field, operator
from jaxdf.discretization import Coordinate, UniformField

from jwave import geometry


def _base_pml(
    transform_fun: Callable, medium: geometry.Medium, exponent=2.0, alpha_max=2.0
) -> jnp.ndarray:
    def pml_edge(x):
        return x / 2 - medium.pml_size

    delta_pml = list(map(pml_edge, medium.domain.N))
    domain = geometry.Domain(N=medium.domain.N, dx=tuple([1.0] * len(medium.domain.N)))
    delta_pml, delta_pml_f = UniformField(domain, len(delta_pml)).from_scalar(
        jnp.asarray(delta_pml), "delta_pml"
    )
    coordinate_discr = Coordinate(domain)
    X = Field(coordinate_discr, params={}, name="X")

    @operator()
    def X_pml(X, delta_pml):
        diff = (jops.elementwise(jnp.abs)(X) + (-1.0) * delta_pml) / (medium.pml_size)
        on_pml = jops.elementwise(lambda x: jnp.where(x > 0, x, 0))(diff)
        alpha = alpha_max * (on_pml ** exponent)
        exp_val = transform_fun(alpha)
        return exp_val

    outfield = X_pml(X=X, delta_pml=delta_pml_f)
    global_params = outfield.get_global_params()
    pml_val = outfield.get_field_on_grid(0)(
        global_params, {"X": {}, "delta_pml": delta_pml}
    )
    return pml_val


def complex_pml_on_grid(
    medium: geometry.Medium, omega: float, exponent=2.0, alpha_max=2.0
) -> jnp.ndarray:
    transform_fun = lambda alpha: 1.0 / (1 + 1j * alpha)
    return _base_pml(transform_fun, medium, exponent, alpha_max)


def td_pml_on_grid(
    medium: geometry.Medium, dt: float, exponent=4.0, alpha_max=1.0, c0=1.0, dx=1.0
) -> jnp.ndarray:
    transform_fun = jops.elementwise(
        lambda alpha: jnp.exp((-1) * alpha * dt * c0 / 2 / dx)
    )
    return _base_pml(transform_fun, medium, exponent, alpha_max)
