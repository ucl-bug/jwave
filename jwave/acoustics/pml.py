from typing import Callable
from jaxdf import Field
from jaxdf.operators.functions import compose
from jwave.geometry import Medium, Domain
from jax import numpy as jnp

def _base_pml(
  transform_fun: Callable,
  medium: Medium,
  exponent: 2.0,
  alpha_max: 2.0
) -> Field:
  def pml_edge(x):
    return x / 2 - medium.pml_size
  
  delta_pml = jnp.asarray(list(map(pml_edge, medium.domain.N)))
  coord_grid = Domain(N=medium.domain.N, dx=tuple([1.0] * len(medium.domain.N))).grid
  
  def _pml_fun(x, delta_pml):
    diff = (jnp.abs(x) - 1.0 * delta_pml) / medium.pml_size
    on_pml = jnp.where(diff > 0, diff, 0)
    alpha = alpha_max * (on_pml ** exponent)
    exp_val = transform_fun(alpha)
    return exp_val

  return _pml_fun(coord_grid, delta_pml)


def complex_pml_on_grid(
    medium: Medium, omega: float, exponent=2.0, alpha_max=2.0
) -> jnp.ndarray:
    transform_fun = lambda alpha: 1.0 / (1 + 1j * alpha)
    return _base_pml(transform_fun, medium, exponent, alpha_max)


def td_pml_on_grid(
    medium: Medium, dt: float, exponent=4.0, alpha_max=1.0, c0=1.0, dx=1.0
) -> jnp.ndarray:
    transform_fun = lambda alpha: jnp.exp((-1) * alpha * dt * c0 / 2 / dx)
    return _base_pml(transform_fun, medium, exponent, alpha_max)
