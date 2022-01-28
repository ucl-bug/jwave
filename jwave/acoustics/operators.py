from unittest.mock import NonCallableMagicMock
from jaxdf import operator, Field
from jaxdf.discretization import OnGrid, FourierSeries
from jaxdf.operators import gradient, diag_jacobian, sum_over_dims
from jwave.geometry import Medium
from numbers import Number
from .pml import complex_pml_on_grid

@operator
def laplacian_with_pml(
  u: OnGrid,
  medium: Medium,
  omega: float,
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
  u: FourierSeries,
  medium: Medium,
  omega: float,
  params = None
):
  # Initialize pml parameters if not provided
  if params == None:
    params = {
      'pml_on_grid': u.replace_params(complex_pml_on_grid(medium, omega)),
      'fft_params': gradient(u)._op_params
    }
    
  pml = params['pml_on_grid']
  fft_params = params['fft_params']
  
  # Making laplacian
  grad_u = gradient(u, params=fft_params)
  mod_grad_u = grad_u*pml
  mod_diag_jacobian = diag_jacobian(mod_grad_u, params=fft_params) * pml
  nabla_u = sum_over_dims(mod_diag_jacobian)
  
  # Density term
  rho0 = medium.density
  if not(issubclass(type(rho0), Field)):
    # Assume it is a number
    rho_u = 0.
  else:
    grad_rho0 = gradient(rho0, params=fft_params)
    rho_u = sum_over_dims(mod_grad_u * grad_rho0) / rho0
  
  # Put everything together
  return nabla_u - rho_u, params

@operator
def wavevector(
  u: Field,
  medium: Medium,
  omega: float,
  params = None
):
  """
  Calculate the wavevector field.
  """
  c = medium.sound_speed
  alpha = medium.attenuation
  k_mod = (omega / c) ** 2 + 2j * (omega ** 3) * alpha / c
  return u * k_mod, None

@operator
def helmholtz(
  u: Field,
  medium: Medium,
  omega: float = 1.0,
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
  omega: float = 1.0,
  params = None
):
  if params == None:
    params = laplacian_with_pml(u, medium, omega)._op_params
  
  # Get the modified laplacian
  L = laplacian_with_pml(u, medium, omega, params=params)

  # Add the wavenumber term
  k = wavevector(u, medium, omega)
  return L + k, params
  
