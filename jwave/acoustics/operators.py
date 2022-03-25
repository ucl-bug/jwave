from jax import numpy as jnp
from jaxdf import Field, operator
from jaxdf.discretization import (
    Continuous,
    FiniteDifferences,
    FourierSeries,
    OnGrid,
)
from jaxdf.operators import compose, diag_jacobian, gradient, sum_over_dims

from jwave.geometry import Medium

from .conversion import db2neper
from .pml import complex_pml, complex_pml_on_grid


@operator
def laplacian_with_pml(
  u: Continuous,
  medium: Medium,
  omega = 1.0,
  params = None
):
  # Initialize coordinate filed
  x = Continuous(None, u.domain, lambda p, x: x)

  # PML Filed
  pml = complex_pml(x, medium, omega)

  # Modified laplacian
  grad_u = gradient(u)
  mod_grad_u = grad_u*pml
  mod_diag_jacobian = diag_jacobian(mod_grad_u)*pml
  return sum_over_dims(mod_diag_jacobian), None

@operator
def laplacian_with_pml(
  u: OnGrid,
  medium: Medium,
  omega = 1.0,
  params = None
):
  pml_grid = complex_pml_on_grid(medium, omega)
  pml = u.replace_params(pml_grid)

  # Making laplacian
  grad_u = gradient(u)
  mod_grad_u = grad_u*pml
  mod_diag_jacobian = diag_jacobian(mod_grad_u) * pml
  nabla_u = sum_over_dims(mod_diag_jacobian)

  # Density term
  rho0 = medium.density
  if not(issubclass(type(rho0), Field)):
    # Assume it is a number
    rho_u = 0.
  else:
    grad_rho0 = gradient(rho0)
    rho_u = sum_over_dims(mod_grad_u * grad_rho0) / rho0

  # Put everything together
  return nabla_u - rho_u, None

@operator
def laplacian_with_pml(
  u: FiniteDifferences,
  medium: Medium,
  omega = 1.0,
  accuracy = 4,
  params = None
):
  rho0 = medium.density
  if params == None:
    params = {
      'pml_on_grid': u.replace_params(complex_pml_on_grid(medium, omega)),
      'stencils':  {
        'gradient': gradient(u, accuracy=accuracy, staggered='forward')._op_params,
        'diag_jacobian': gradient(u, accuracy=accuracy, staggered='backward')._op_params,
      }}

  pml = params['pml_on_grid']
  stencils = params['stencils']

  # Making laplacian
  grad_u = gradient(u, stencils['gradient'])
  mod_grad_u = grad_u*pml
  mod_diag_jacobian = diag_jacobian(mod_grad_u, stencils['diag_jacobian'])
  nabla_u = sum_over_dims(mod_diag_jacobian * pml)

  if not(issubclass(type(rho0), Field)):
    # Assume it is a number
    rho_u = 0.
  else:
    if not('fft_rho0' in params.keys()):
      params['fft_rho0'] = gradient(rho0)._op_params

    def _axis_dx(rho0, axis):
      su = jnp.roll(rho0, -1, axis)
      g_rho0 = (su - rho0) / u.domain.dx[axis]
      return g_rho0

    def grad_density(rho0):
      rho0 = rho0.params[...,0]
      g_rho0 = jnp.stack([_axis_dx(rho0, axis) for axis in range(rho0.ndim)],-1)
      g_rho0 = u.replace_params(g_rho0)
      return g_rho0

    grad_rho0 = gradient(rho0, params=params['fft_rho0'])
    #grad_rho0 = grad_density(rho0)
    rho_u = sum_over_dims(mod_grad_u * grad_rho0) / rho0

  return nabla_u, params


@operator
def laplacian_with_pml(
  u: FourierSeries,
  medium: Medium,
  omega = 1.0,
  params = None
):
  rho0 = medium.density

  # Initialize pml parameters if not provided
  if params == None:
    params = {
      'pml_on_grid': u.replace_params(complex_pml_on_grid(medium, omega)),
      'fft_u':  gradient(u)._op_params,
    }

  pml = params['pml_on_grid']

  # Making laplacian
  grad_u = gradient(u, params=params['fft_u'])
  mod_grad_u = grad_u*pml
  mod_diag_jacobian = diag_jacobian(mod_grad_u, params=params['fft_u']) * pml
  nabla_u = sum_over_dims(mod_diag_jacobian)

  # Density term
  if not(issubclass(type(rho0), Field)):
    # Assume it is a number
    rho_u = 0.
  else:
    if not('fft_rho0' in params.keys()):
      params['fft_rho0'] = gradient(rho0)._op_params

    def _axis_dx(rho0, axis):
      su = jnp.roll(rho0, -1, axis)
      g_rho0 = (su - rho0) / u.domain.dx[axis]
      return g_rho0

    def grad_density(rho0):
      rho0 = rho0.params[...,0]
      g_rho0 = jnp.stack([_axis_dx(rho0, axis) for axis in range(rho0.ndim)],-1)
      g_rho0 = u.replace_params(g_rho0)
      return g_rho0

    grad_rho0 = gradient(rho0, params=params['fft_rho0'])
    #grad_rho0 = grad_density(rho0)
    rho_u = sum_over_dims(mod_grad_u * grad_rho0) / rho0

  # Put everything together
  return nabla_u - rho_u, params

@operator
def wavevector(
  u: Field,
  medium: Medium,
  omega = 1.0,
  params = None
):
  """
  Calculate the wavevector field.
  """
  c = medium.sound_speed
  alpha = medium.attenuation
  trans_fun = lambda x: db2neper(x, 2.)
  alpha = compose(alpha)(trans_fun)
  k_mod = (omega / c) ** 2 + 2j * (omega ** 3) * alpha / c
  return u * k_mod, None


@operator
def helmholtz(
  u: FiniteDifferences,
  medium: Medium,
  omega = 1.0,
  params = None
):
  if params == None:
    params = laplacian_with_pml(u, medium, omega)._op_params

  # Get the modified laplacian
  L = laplacian_with_pml(u, medium, omega)

  # Add the wavenumber term
  k = wavevector(u, medium, omega)
  return L + k, params

@operator
def helmholtz(
  u: Field,
  medium: Medium,
  omega = 1.0,
  params = None
):
  # Get the modified laplacian
  L = laplacian_with_pml(u, medium, omega)

  # Add the wavenumber term
  k = wavevector(u, medium, omega)
  return L + k, None


@operator
def helmholtz(
  u: FourierSeries,
  medium: Medium,
  omega = 1.0,
  params = None
):
  if params == None:
    params = laplacian_with_pml(u, medium, omega)._op_params

  # Get the modified laplacian
  L = laplacian_with_pml(u, medium, omega, params=params)

  # Add the wavenumber term
  k = wavevector(u, medium, omega)
  return L + k, params
