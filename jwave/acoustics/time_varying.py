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


def _get_kspace_op(domain, c_ref, dt):
  # Get the frequency axis manually, since we
  # are nor using the rFFT
  # TODO: Implement operators with rFFT
  def f(N, dx):
    return jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
  k_vec = [f(n, delta) for n, delta in zip(domain.N, domain.dx)]

  # Building k-space operator
  K = jnp.stack(jnp.meshgrid(*k_vec, indexing='ij'))
  k_magnitude = jnp.sqrt(jnp.sum(K ** 2, 0))
  k_space_op = jnp.sinc(c_ref * k_magnitude * dt / (2 * jnp.pi))
  parameters = {"k_vec": k_vec, "k_space_op": k_space_op}
  return parameters

def _shift_rho(rho0, direction, dx):
  if isinstance(rho0, OnGrid):
    rho0_params = rho0.params[...,0]
    def linear_interp(u, axis):
      return 0.5*(jnp.roll(u, -direction, axis) + u)
    rho0 = jnp.stack([linear_interp(rho0_params, n) for n in range(rho0.ndim)], axis=-1)
  elif isinstance(rho0, Field):
    rho0 = shift_operator(rho0, direction*dx)
  else:
    pass
  return rho0

@operator
def momentum_conservation_rhs(
  p: OnGrid,
  u: OnGrid,
  medium: Medium,
  c_ref = 1.0,
  dt = 1.0,
  params = None
) -> OnGrid:
  r"""Staggered implementation of the momentum conservation equation.

  Args:
    p (OnGrid): The pressure field.
    u (OnGrid): The velocity field.
    medium (Medium): The medium.
    c_ref (float): The reference sound speed. **Unused**
    dt (float): The time step. **Unused**
    params: The operator parameters. **Unused**

  Returns:
    OnGrid: The right hand side of the momentum conservation equation.
  """
  # Staggered implementation
  dx = np.asarray(u.domain.dx)
  rho0 = _shift_rho(medium.density, 1, dx)
  dp = diag_jacobian(p, stagger = [0.5])
  return -dp / rho0, params


@operator
def momentum_conservation_rhs(
  p: FourierSeries,
  u: FourierSeries,
  medium: Medium,
  c_ref = 1.0,
  dt = 1.0,
  params = None
) -> FourierSeries:
  r"""Staggered implementation of the momentum conservation equation.

  Args:
    p (FourierSeries): The pressure field.
    u (FourierSeries): The velocity field.
    medium (Medium): The medium.
    c_ref (float): The reference sound speed. Used to calculate the k-space operator.
    dt (float): The time step. Used to calculate the k-space operator.
    params: The operator parameters.

  Returns:
    FourierSeries: The right hand side of the momentum conservation equation.
  """
  if params == None:
    params = _get_kspace_op(p.domain, c_ref, dt)

  dx = np.asarray(u.domain.dx)
  direction = 1

  # Shift rho
  rho0 = _shift_rho(medium.density, direction, dx)

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
  p: OnGrid,
  u: OnGrid,
  mass_source: object,
  medium: Medium,
  c_ref,
  dt,
  params = None
) -> OnGrid:
  r"""Implementation of the mass conservation equation. The pressure field
  is assumed to be staggered forward for each component, and will be staggered
  backward before being multiplied by the ambient density.

  Args:
    p (OnGrid): The pressure field.
    u (OnGrid): The velocity field.
    mass_source (object): The mass source.
    medium (Medium): The medium.
    c_ref (float): The reference sound speed. **Unused**
    dt (float): The time step. **Unused**
    params: The operator parameters. **Unused**

  Returns:
    OnGrid: The right hand side of the mass conservation equation.
  """

  rho0 = medium.density
  c0 = medium.sound_speed
  dx = np.asarray(p.domain.dx)

  # Staggered implementation
  du = diag_jacobian(u, stagger=[-0.5])
  update = -du * rho0 + 2 * mass_source / (c0 * p.ndim * dx)
  return update, params



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
  r"""Implementation of the mass conservation equation. The pressure field
  is assumed to be staggered forward for each component, and will be staggered
  backward before being multiplied by the ambient density.

  Args:
    p (FourierSeries): The pressure field.
    u (FourierSeries): The velocity field.
    mass_source (object): The mass source.
    medium (Medium): The medium.
    c_ref (float): The reference sound speed. Used to calculate the k-space operator.
    dt (float): The time step. Used to calculate the k-space operator.
    params: The operator parameters.

  Returns:
    FourierSeries: The right hand side of the mass conservation equation.
  """

  if params == None:
    params = _get_kspace_op(p.domain, c_ref, dt)

  dx = np.asarray(p.domain.dx)
  direction = -1

  k_vec = params['k_vec']
  k_space_op = params['k_space_op']
  rho0 = medium.density
  c0 = medium.sound_speed

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
  update = -p.replace_params(du) * rho0 + 2 * mass_source / (c0 * p.ndim * dx)

  return update, params

@operator
def pressure_from_density(
  rho: Field,
  medium: Medium,
  params = None
) -> Field:
  r"""Compute the pressure field from the density field.

  Args:
    rho (Field): The density field.
    medium (Medium): The medium.
    params: The operator parameters. **Unused**

  Returns:
    Field: The pressure field.
  """
  rho_sum = sum_over_dims(rho)
  c0 = medium.sound_speed
  return (c0**2) * rho_sum, params

OnGridOrScalars = Union[
  MediumObject[object,object,OnGrid],
  MediumObject[object,OnGrid,object],
  MediumObject[OnGrid,object,object],
]

def ongrid_wave_prop_params(
  medium: OnGrid,
  time_axis: TimeAxis,
  *args,
  **kwargs,
):
  # Check which elements of medium are a field
  x = [x for x in [medium.sound_speed, medium.density, medium.attenuation] if isinstance(x, Field)][0]

  dt = time_axis.dt
  c_ref = functional(medium.sound_speed)(jnp.amax)

  # Making PML on grid for rho and u
  def make_pml(staggering=0.0):
    pml_grid = td_pml_on_grid(
      medium,
      dt,
      c0=c_ref,
      dx=medium.domain.dx[0],
      coord_shift=staggering
    )
    pml = x.replace_params(pml_grid)
    return pml

  pml_rho = make_pml()
  pml_u = make_pml(staggering=0.5)

  return {
    'pml_rho': pml_rho,
    'pml_u': pml_u,
  }

@operator
def simulate_wave_propagation(
  medium: OnGridOrScalars,
  time_axis: TimeAxis,
  *,
  sources = None,
  sensors = None,
  u0 = None,
  p0 = None,
  checkpoint: bool = False,
  smooth_initial = True,
  params = None
):
  r"""Simulate the wave propagation operator.

  Args:
    medium (OnGridOrScalars): The medium.
    time_axis (TimeAxis): The time axis.
    sources (Any): The source terms. It can be any jax traceable object that
      implements the method `sources.on_grid(n)`, which returns the source
      field at the nth time step.
    sensors (Any): The sensor terms. It can be any jax traceable object that
      can be called as `sensors(p,u,rho)`, where `p` is the pressure field,
      `u` is the velocity field, and `rho` is the density field. The return
      value of this function is the recorded field. If `sensors` is not
      specified, the recorded field is the entire pressure field.
    u0 (Field): The initial velocity field. If `None`, the initial velocity
      field is set depending on the `p0` value, such that `u(t=0)=0`. Note that
      the velocity field is staggered forward by half time step relative to the
      pressure field.
    p0 (Field): The initial pressure field. If `None`, the initial pressure
      field is set to zero.
    checkpoint (bool): Whether to checkpoint the simulation at each time step.
      See [jax.checkpoint](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)
    smooth_initial (bool): Whether to smooth the initial conditions.
    params: The operator parameters.

  Returns:
    Any: The recording of the sensors at each time step.
  """

  # Default sensors simply return the presure field
  if sensors is None:
    sensors = lambda p, u, rho: p

  # Setup parameters
  output_steps = jnp.arange(0, time_axis.Nt, 1)
  dt = time_axis.dt
  c_ref = functional(medium.sound_speed)(jnp.amax)

  if params == None:
    params = ongrid_wave_prop_params(medium, time_axis)

  # Get parameters
  pml_rho = params['pml_rho']
  pml_u = params['pml_u']

  # Initialize variables
  shape = tuple(list(medium.domain.N) + [len(medium.domain.N),])
  shape_one = tuple(list(medium.domain.N) + [1,])
  if u0 is None:
    u0 = pml_u.replace_params(jnp.zeros(shape))
  else:
    assert u0.dim == len(medium.domain.N)
  if p0 is None:
    p0 = pml_rho.replace_params(jnp.zeros(shape_one))
  else:
    if smooth_initial:
      p0_params = p0.params[...,0]
      p0_params = jnp.expand_dims(smooth(p0_params), -1)
      p0 = p0.replace_params(p0_params)

    # Force u(t=0) to be zero accounting for time staggered grid
    u0 = -dt * momentum_conservation_rhs(p0, u0, medium, c_ref, dt) / 2

  # Initialize acoustic density
  rho = p0.replace_params(
      jnp.stack([p0.params[...,i] for i in range(p0.ndim)], axis=-1)
    ) / p0.ndim
  rho = rho/(medium.sound_speed ** 2)

  # define functions to integrate
  fields = [p0, u0, rho]

  def scan_fun(fields, n):
    p, u, rho = fields
    if sources is None:
      mass_src_field = 0.0
    else:
      mass_src_field = sources.on_grid(n)

    du = momentum_conservation_rhs(p, u, medium, c_ref, dt)
    u = pml_u*(pml_u*u + dt * du)

    drho = mass_conservation_rhs(p, u, mass_src_field, medium, c_ref, dt)
    rho = pml_rho*(pml_rho*rho + dt * drho)

    p = pressure_from_density(rho, medium)
    return [p, u, rho], sensors(p,u,rho)

  if checkpoint:
    scan_fun = jax.checkpoint(scan_fun)

  _, ys = jax.lax.scan(scan_fun, fields, output_steps)

  return ys

# Custom type
FourierOrScalars = Union[
  MediumObject[object,object,FourierSeries],
  MediumObject[object,FourierSeries,object],
  MediumObject[FourierSeries,object,object],
  MediumObject[object,object,object],
]

def fourier_wave_prop_params(
  medium: FourierOrScalars,
  time_axis: TimeAxis,
  *args,
  **kwargs,
):
  dt = time_axis.dt
  c_ref = functional(medium.sound_speed)(jnp.amax)

  # Making PML on grid for rho and u
  def make_pml(staggering=0.0):
    pml_grid = td_pml_on_grid(
      medium,
      dt,
      c0=c_ref,
      dx=medium.domain.dx[0],
      coord_shift=staggering
    )
    pml = FourierSeries(pml_grid, medium.domain)
    return pml

  pml_rho = make_pml()
  pml_u = make_pml(staggering=0.5)

  # Get k-space operator
  fourier = _get_kspace_op(medium.domain, c_ref, dt)

  return {
    'pml_rho': pml_rho,
    'pml_u': pml_u,
    'fourier': fourier,
  }

@operator(init_params=fourier_wave_prop_params)
def simulate_wave_propagation(
  medium: FourierOrScalars,
  time_axis: TimeAxis,
  *,
  sources = None,
  sensors = None,
  u0 = None,
  p0 = None,
  checkpoint: bool = False,
  smooth_initial = True,
  params = None
):
  r"""Simulates the wave propagation operator using the PSTD method. This
  implementation is equivalent to the `kspaceFirstOrderND` function in the
  k-Wave Toolbox.

  Args:
    medium (FourierOrScalars): The medium.
    time_axis (TimeAxis): The time axis.
    sources (Any): The source terms. It can be any jax traceable object that
      implements the method `sources.on_grid(n)`, which returns the source
      field at the nth time step.
    sensors (Any): The sensor terms. It can be any jax traceable object that
      can be called as `sensors(p,u,rho)`, where `p` is the pressure field,
      `u` is the velocity field, and `rho` is the density field. The return
      value of this function is the recorded field. If `sensors` is not
      specified, the recorded field is the entire pressure field.
    u0 (Field): The initial velocity field. If `None`, the initial velocity
      field is set depending on the `p0` value, such that `u(t=0)=0`. Note that
      the velocity field is staggered forward by half time step relative to the
      pressure field.
    p0 (Field): The initial pressure field. If `None`, the initial pressure
      field is set to zero.
    checkpoint (bool): Whether to checkpoint the simulation at each time step.
      See [jax.checkpoint](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)
    smooth_initial (bool): Whether to smooth the initial conditions.
    params: The operator parameters.

  Returns:
    Any: The recording of the sensors at each time step.
  """
  # Default sensors simply return the presure field
  if sensors is None:
    sensors = lambda p, u, rho: p

  # Setup parameters
  output_steps = jnp.arange(0, time_axis.Nt, 1)
  dt = time_axis.dt
  c_ref = functional(medium.sound_speed)(jnp.amax)
  if params == None:
    params = fourier_wave_prop_params(medium, time_axis)

  # Get parameters
  pml_rho = params['pml_rho']
  pml_u = params['pml_u']

  # Initialize variables
  shape = tuple(list(medium.domain.N) + [len(medium.domain.N),])
  shape_one = tuple(list(medium.domain.N) + [1,])
  if u0 is None:
    u0 = pml_u.replace_params(jnp.zeros(shape))
  else:
    assert u0.dim == len(medium.domain.N)
  if p0 is None:
    p0 = pml_rho.replace_params(jnp.zeros(shape_one))
  else:
    if smooth_initial:
      p0_params = p0.params[...,0]
      p0_params = jnp.expand_dims(smooth(p0_params), -1)
      p0 = p0.replace_params(p0_params)

    # Force u(t=0) to be zero accounting for time staggered grid
    u0 = -dt * momentum_conservation_rhs(p0, u0, medium, c_ref, dt, params=params['fourier']) / 2

  # Initialize acoustic density
  rho = p0.replace_params(
      jnp.stack([p0.params[...,i] for i in range(p0.ndim)], axis=-1)
    ) / p0.ndim
  rho = rho/(medium.sound_speed ** 2)

  # define functions to integrate
  fields = [p0, u0, rho]

  def scan_fun(fields, n):
    p, u, rho = fields
    if sources is None:
      mass_src_field = 0.0
    else:
      mass_src_field = sources.on_grid(n)

    du = momentum_conservation_rhs(p, u, medium, c_ref, dt, params=params['fourier'])
    u = pml_u*(pml_u*u + dt * du)

    drho = mass_conservation_rhs(p, u, mass_src_field, medium, c_ref, dt, params=params['fourier'])
    rho = pml_rho*(pml_rho*rho + dt * drho)

    p = pressure_from_density(rho, medium)
    return [p, u, rho], sensors(p,u,rho)

  if checkpoint:
    scan_fun = jax.checkpoint(scan_fun)

  _, ys = jax.lax.scan(scan_fun, fields, output_steps)

  return ys


if __name__ == "__main__":
    pass
