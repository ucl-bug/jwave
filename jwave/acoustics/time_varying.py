from typing import Union

import jax
import numpy as np
from jax import numpy as jnp
from jaxdf import Field, operator
from jaxdf.discretization import FourierSeries, OnGrid
from jaxdf.operators import (
    diag_jacobian,
    functional,
    shift_operator,
    sum_over_dims,
)

from jwave.geometry import Medium, MediumObject, TimeAxis
from jwave.signal_processing import smooth

from .pml import td_pml_on_grid


def _get_kspace_op(field, c_ref, dt):
  # Get the frequency axis manually, since we
  # are nor using the rFFT
  # TODO: Implement operators with rFFT
  def f(N, dx):
    return jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
  k_vec = [f(n, delta) for n, delta in zip(field.domain.N, field.domain.dx)]

  # Building k-space operator
  K = jnp.stack(jnp.meshgrid(*k_vec, indexing='ij'))
  k_magnitude = jnp.sqrt(jnp.sum(K ** 2, 0))
  k_space_op = jnp.sinc(c_ref * k_magnitude * dt / (2 * jnp.pi))
  parameters = {"k_vec": k_vec, "k_space_op": k_space_op}
  return parameters

def _shift_rho_for_fourier(rho0, direction, dx):
  if isinstance(rho0, OnGrid):
    rho0_params = rho0.params[...,0]
    def linear_interp(u, axis):
      return 0.5*(jnp.roll(u, direction, axis) + u)
    rho0 = jnp.stack([linear_interp(rho0_params, n) for n in range(rho0.ndim)], axis=-1)
  elif isinstance(rho0, Field):
    rho0 = shift_operator(rho0, direction*dx)
  else:
    pass
  return rho0

@operator
def momentum_conservation_rhs(
  p: OnGrid,
  rho0: object,
  params = None
) -> OnGrid:
  # Staggered implementation
  dx = np.asarray(p.domain.dx)/2
  dp = shift_operator(diag_jacobian(p), dx)
  return -dp / rho0, params

@operator
def momentum_conservation_rhs(
  p: OnGrid,
  rho0: OnGrid,
  params = None
) -> OnGrid:
  # Staggered implementation
  dx = np.asarray(p.domain.dx)/2
  dp = shift_operator(diag_jacobian(p), dx)
  rho0 = shift_operator(rho0, dx)
  return -dp / rho0, params


@operator
def momentum_conservation_rhs(
  p: FourierSeries,
  u: FourierSeries,
  medium: Medium,
  c_ref,
  dt,
  params = None
) -> FourierSeries:

  if params == None:
    params = _get_kspace_op(p, c_ref, dt)

  dx = np.asarray(u.domain.dx)
  direction = 1

  # Shift rho
  rho0 = _shift_rho_for_fourier(medium.density, direction, dx)

  # Take a shifted gradient of the pressure
  k_vec = params['k_vec']
  k_space_op = params['k_space_op']

  shift_and_k_op = [
    1j * k * jnp.exp(1j * k * direction * delta / 2)
    for k, delta in zip(k_vec, dx)
  ]

  p_params = p.params[...,0]
  Fu = jnp.fft.fftn(p_params)

  def single_grad(axis):
    Fx = jnp.moveaxis(Fu, axis, -1)
    k_op = jnp.moveaxis(k_space_op, axis, -1)
    iku = jnp.moveaxis(Fx * shift_and_k_op[axis] * k_op, -1, axis)
    return jnp.fft.ifftn(iku).real

  dp = jnp.stack([single_grad(i) for i in range(p.ndim)], axis=-1)
  update = -p.replace_params(dp) / rho0

  return update, params

@operator
def mass_conservation_rhs(
  u: Field,
  rho0: Field,
  mass_source: Field,
  params = None
) -> Field:
  return - rho0 * diag_jacobian(u) + mass_source, params


@operator
def mass_conservation_rhs(
  u: OnGrid,
  rho0: object,
  mass_source: object,
  params = None
) -> Field:
  # Staggered implementation
  du = diag_jacobian(u)
  return - rho0 * du + mass_source, params


@operator
def mass_conservation_rhs(
  u: OnGrid,
  rho0: OnGrid,
  mass_source: object,
  params = None
) -> Field:
  # Staggered implementation
  dx = -np.asarray(u.domain.dx)/2
  du = shift_operator(diag_jacobian(u), dx)
  return - rho0 * du + mass_source, params



@operator
def mass_conservation_rhs(
  p: FourierSeries,
  u: FourierSeries,
  mass_source: object,
  medium: Medium,
  c_ref,
  dt,
  params = None
) -> FourierSeries:

  if params == None:
    params = _get_kspace_op(p, c_ref, dt)

  dx = np.asarray(u.domain.dx)
  direction = -1

  k_vec = params['k_vec']
  k_space_op = params['k_space_op']
  rho0 = medium.density

  # Add shift to k vector
  shift_and_k_op = [
    1j * k * jnp.exp(1j * k * direction * delta / 2)
    for k, delta in zip(k_vec, dx)
  ]

  def single_grad(axis, u):
    Fu = jnp.fft.fftn(u)
    Fx = jnp.moveaxis(Fu, axis, -1)
    k_op = jnp.moveaxis(k_space_op, axis, -1)
    iku = jnp.moveaxis(Fx * shift_and_k_op[axis] *  k_op, -1, axis)
    return jnp.fft.ifftn(iku).real

  du = jnp.stack([single_grad(i, u.params[...,i]) for i in range(p.ndim)], axis=-1)
  update = -p.replace_params(du) * rho0 + mass_source / du.ndim

  return update, params

@operator
def pressure_from_density(
  rho: Field,
  medium: Medium,
  params = None
) -> Field:
  rho_sum = sum_over_dims(rho)
  c0 = medium.sound_speed
  return (c0**2) * rho_sum, params

# Custom type
FourierOrScalars = Union[
  MediumObject[object,object,FourierSeries],
  MediumObject[object,FourierSeries,object],
  MediumObject[FourierSeries,object,object],
  MediumObject[object,object,object],
]

@operator
def simulate_wave_propagation(
  medium: FourierOrScalars,
  time_axis: TimeAxis,
  sources = None,
  sensors = None,
  u0 = None,
  p0 = None,
  checkpoint: bool = False,
  params = None,
  smooth_initial = True,
):

  # Default sensors simply return the presure field
  if sensors is None:
    sensors = lambda p, u, rho: p

  # Setup parameters
  if params == None:
    c_ref = functional(medium.sound_speed)(jnp.amax)
    dt = time_axis.dt

    t = jnp.arange(0, time_axis.t_end + time_axis.dt, time_axis.dt)
    output_steps = (t / dt).astype(jnp.int32)

    # Making PML on grid
    pml_grid = td_pml_on_grid(medium, dt, c0=c_ref, dx=medium.domain.dx[0])
    if issubclass(type(pml_grid), Field):
      pml = medium.sound_speed.replace_params(pml_grid)
    else:
      pml = FourierSeries(pml_grid, medium.domain)

    # Get k-space operator
    fourier = _get_kspace_op(pml, c_ref, dt)
    params = {
      'pml': pml,
      'output_steps': output_steps,
      'fourier': fourier,
    }

  # Get parameters
  pml = params['pml']

  # Initialize variables
  shape = tuple(list(medium.domain.N) + [len(medium.domain.N),])
  shape_one = tuple(list(medium.domain.N) + [1,])
  if u0 is None:
    u0 = pml.replace_params(jnp.zeros(shape))
  else:
    assert u0.dim == len(medium.domain.N)
  if p0 is None:
    p0 = pml.replace_params(jnp.zeros(shape_one))
  else:
    if smooth_initial:
      p0_params = p0.params[...,0]
      p0_params = jnp.expand_dims(smooth(p0_params), -1)
      p0 = p0.replace_params(p0_params)

  # Initialize acoustic density
  rho = p0.replace_params(
      jnp.stack([p0.params[...,i] for i in range(p0.ndim)], axis=-1)
    ) / p0.ndim
  rho = rho/(medium.sound_speed ** 2)

  # define functions to integrate
  fields = [p0, u0, rho]
  alpha = params['pml']
  output_steps = params['output_steps']

  def scan_fun(fields, n):
    p, u, rho = fields
    if sources is None:
      mass_src_field = 0.0
    else:
      mass_src_field = sources.on_grid(n)

    du = momentum_conservation_rhs(p, u, medium, c_ref, dt, params=params['fourier'])
    u = alpha*(alpha*u + dt * du)

    drho = mass_conservation_rhs(p, u, mass_src_field, medium, c_ref, dt, params=params['fourier'])
    rho = alpha*(alpha*rho + dt * drho)

    p = pressure_from_density(rho, medium)
    return [p, u, rho], sensors(p,u,rho)

  if checkpoint:
    scan_fun = jax.checkpoint(scan_fun)

  _, ys = jax.lax.scan(scan_fun, fields, output_steps)

  return ys


if __name__ == "__main__":
    pass
