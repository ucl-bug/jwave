from typing import Sequence

import jax
import numpy as np
from jax import numpy as jnp
from jaxdf import Field, operator
from jaxdf.discretization import FiniteDifferences, FourierSeries, OnGrid
from jaxdf.operators import (
    diag_jacobian,
    gradient,
    shift_operator,
    sum_over_dims,
)
from jaxdf.operators.differential import _get_ffts

from jwave.geometry import Medium, TimeAxis

from .pml import td_pml_on_grid


@operator
def momentum_conservation_rhs(
  p: Field,
  rho0: Field,
  params = None
) -> Field:
  return - gradient(p) / rho0

@operator
def momentum_conservation_rhs(
  p: OnGrid,
  rho0: OnGrid,
  params = None
) -> OnGrid:
  # Staggered implementation
  dx = np.asarray(p.domain.dx/2)
  dp = shift_operator(gradient(p), dx)
  rho0 = shift_operator(rho0, dx)
  return -dp / rho0

@operator
def momentum_conservation_rhs(
  p: FourierSeries,
  rho0: FiniteDifferences,
  params = None
) -> FourierSeries:

  if params == None:
    params = {'k_vec': p._freq_axis}

  # Shift rho
  rho0_params = rho0.params[...,0]
  def linear_interp(u, axis):
    return 0.5*(jnp.roll(u, 1, axis) + u)
  rho0 = jnp.stack([linear_interp(rho0_params, n) for n in range(rho0.ndim)], axis=-1)

  # Take a shifted gradient of the pressure
  assert p.dims == 1 # Gradient only defined for scalar fields

  ffts = _get_ffts(p)
  k_vec = params['k_vec']
  u = p.params[...,0]

  def single_grad(axis, u):
    u = jnp.moveaxis(u, axis, -1)
    Fx = ffts[0](u, axis=-1)
    iku = 1j * Fx * k_vec[axis]
    du = ffts[1](iku, axis=-1, n=u.shape[-1])
    return jnp.moveaxis(du, -1, axis)

  dp = jnp.stack([single_grad(i, u) for i in range(p.ndim)], axis=-1)

  # Combine and shift back
  update = -dp / rho0




@operator
def mass_conservation_rhs(
  u: Field,
  rho0: Field,
  mass_source: Field,
  params = None
) -> Field:
  return - rho0 * diag_jacobian(u) + mass_source

@operator
def mass_conservation_rhs(
  u: OnGrid,
  rho0: OnGrid,
  mass_source: OnGrid,
  params = None
) -> Field:
  # Staggered implementation
  dx = -np.asarray(u.domain.dx)/2
  du = shift_operator(diag_jacobian(u), dx)
  return - rho0 * du + mass_source

@operator
def pressure_from_density(
  rho: Field,
  c0: Field,
  params = None
) -> Field:
  return - (c0**2) * sum_over_dims(rho)


def _unroll(x):
  def _first_dim_to_list(leave):
    # Only if subclass of field
    if issubclass(type(leave), Field):
      [leave.replace_params(y) for y in leave.params]
    else:
      leave
  return jax.tree_util.tree_map(_first_dim_to_list, x)

@operator
def simulate_wave_equation(
  medium: Medium,
  time_axis: TimeAxis,
  sources = None,
  sensors = None,
  u0 = None,
  p0 = None,
  checkpoint: bool = False,
  params = None
) -> Sequence[Field]:
  # Check that the sound speed is OnGrid
  if not(issubclass(type(medium.sound_speed), OnGrid)):
    raise ValueError('medium.sound_speed must be a subclass of OnGrid')

  if sensors is None:
    sensors = lambda x: x

  # Setup parameters
  if params == None:
    c_ref = jnp.amax(medium.sound_speed.on_grid)
    dt = time_axis.dt

    t = jnp.arange(0, time_axis.t_end + time_axis.dt, time_axis.dt)
    output_steps = (t / dt).astype(jnp.int32)

    # Making PML on grid
    pml_grid = td_pml_on_grid(medium, dt, c0=c_ref, dx=medium.domain.dx[0])
    pml = medium.sound_speed.replace_params(pml_grid)
    params = {'pml': pml,'output_steps': output_steps}

  # Initialize variables
  shape = tuple(list(medium.domain.N) + [len(medium.domain.N),])
  if u0 is None:
    u0 = medium.sound_speed.replace_params(jnp.zeros(shape))
  else:
    assert u0.dim == len(medium.domain.N)
  if p0 is None:
    p0 = medium.sound_speed.replace_params(jnp.zeros(shape))
  else:
    assert p0.dim == len(medium.domain.N)

  # Initialize acoustic density
  rho_params = p0.on_grid/(medium.sound_speed.on_grid ** 2)
  rho_params = jnp.stack([rho_params]*p0.ndim)
  rho = p0.replace_params(rho_params)

  # define functions to integrate
  fields = [u0, rho]
  rho0 = medium.density
  c = medium.sound_speed
  alpha = params['pml']
  output_steps = params['output_steps']

  def scan_fun(fields, n):
    u, rho = fields
    mass_src_field = sources.on_grid(n)

    du = momentum_conservation_rhs(fields[0], rho0)
    u = alpha*(alpha*u + dt * du)

    drho = mass_conservation_rhs(u, rho0, mass_src_field)
    rho = alpha*(alpha*rho + dt * rho)

    p = pressure_from_density(rho, c)

    return [u, rho0], sensors({'u': u, 'rho': rho, 'p': p})

  if checkpoint:
    scan_fun = jax.checkpoint(scan_fun)

  _, ys = jax.lax.scan(scan_fun, fields, output_steps)

  return _unroll(ys)


if __name__ == "__main__":
    pass
