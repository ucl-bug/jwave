from jax import numpy as jnp
from jaxdf import Field, operator
from jaxdf.discretization import (
    Continuous,
    FiniteDifferences,
    FourierSeries,
    OnGrid,
)
from jaxdf.operators import (
    compose,
    diag_jacobian,
    functional,
    gradient,
    shift_operator,
    sum_over_dims,
)

from jwave.geometry import Medium

from .conversion import db2neper
from .pml import complex_pml, complex_pml_on_grid


@operator
def laplacian_with_pml(
  u: Continuous,
  medium: Medium,
  omega = 1.0,
  params = None
) -> Continuous:
  r"""Laplacian operator with PML for `Continuous` complex fields.

  Args:
    u (Continuous): Continuous complex field.
    medium (Medium): Medium object
    omega (float): Angular frequency.
    params (None, optional): Parameters for the operator.

  Returns:
    Continuous: Modified Laplacian operator applied to `u`.
  """
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
) -> OnGrid:
  r"""Laplacian operator with PML for `OnGrid` complex fields.

  Args:
    u (OnGrid): OnGrid complex field.
    medium (Medium): Medium object
    omega (float): Angular frequency.
    params (None, optional): Parameters for the operator.

  Returns:
    OnGrid: Modified Laplacian operator applied to `u`.
  """
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
  params = None
) -> FiniteDifferences:
  r"""Laplacian operator with PML for `FiniteDifferences` complex fields.

  Args:
    u (FiniteDifferences): FiniteDifferences complex field.
    medium (Medium): Medium object
    omega (float): Angular frequency.
    params (None, optional): Parameters for the operator.

  Returns:
    FiniteDifferences: Modified Laplacian operator applied to `u`.
  """
  rho0 = medium.density
  if params == None:
    params = {
      'pml_on_grid': [
        u.replace_params(complex_pml_on_grid(medium, omega, shift= u.domain.dx[0]/2)),
        u.replace_params(complex_pml_on_grid(medium, omega, shift=-u.domain.dx[0]/2))
      ],
      'stencils':  {
        'gradient': gradient.default_params(u, stagger=[0.5]),
        'gradient_unstaggered': gradient.default_params(u),
        'diag_jacobian': diag_jacobian.default_params(u, stagger=[-0.5]),
      }
    }

  pml = params['pml_on_grid']
  stencils = params['stencils']

  # Making laplacian
  grad_u = gradient(u, [0.5], params=stencils['gradient'])
  mod_grad_u = grad_u*pml[0]
  mod_diag_jacobian = diag_jacobian(mod_grad_u, [-0.5], params=stencils['diag_jacobian'])
  nabla_u = sum_over_dims(mod_diag_jacobian * pml[1])

  if not(issubclass(type(rho0), Field)):
    # Assume it is a number
    rho_u = 0.
  else:
    grad_u = gradient(u,params=stencils['gradient_unstaggered'])
    grad_rho0 = gradient(rho0, [0], params=stencils['gradient_unstaggered'])
    rho_u = sum_over_dims(mod_grad_u * grad_rho0) / rho0

  return nabla_u - rho_u, params

@operator
def laplacian_with_pml(
  u: FourierSeries,
  medium: Medium,
  omega = 1.0,
  params = None
) -> FourierSeries:
  r"""Laplacian operator with PML for `FourierSeries` complex fields.

  Args:
    u (FourierSeries): FourierSeries complex field.
    medium (Medium): Medium object
    omega (float): Angular frequency.
    params (None, optional): Parameters for the operator.

  Returns:
    FourierSeries: Modified Laplacian operator applied to `u`.
  """
  rho0 = medium.density

  # Initialize pml parameters if not provided
  if params == None:
    params = {
      'pml_on_grid': [
        u.replace_params(complex_pml_on_grid(medium, omega, shift= u.domain.dx[0]/2)),
        u.replace_params(complex_pml_on_grid(medium, omega, shift=-u.domain.dx[0]/2))
      ],
      'fft_u':  gradient.default_params(u),
    }

  pml = params['pml_on_grid']

  # Making laplacian
  grad_u = gradient(
    u,
    stagger=[0.5],
    correct_nyquist=False,
    params=params['fft_u'])
  mod_grad_u = grad_u*pml[0]
  mod_diag_jacobian = diag_jacobian(
    mod_grad_u,
    stagger=[-0.5],
    correct_nyquist=False,
    params=params['fft_u']
  ) * pml[1]
  nabla_u = sum_over_dims(mod_diag_jacobian)

  # Density term
  if not(issubclass(type(rho0), Field)):
    # Assume it is a number
    rho_u = 0.
  else:
    assert isinstance(rho0, FourierSeries), "rho0 must be a FourierSeries or a number when used with FourierSeries fields"

    if not('fft_rho0' in params.keys()):
      params['fft_rho0'] = gradient.default_params(rho0)

    grad_rho0 = gradient(rho0, stagger=[0.5], params=params['fft_rho0'])
    dx = list(map(lambda x: -x/2, u.domain.dx))
    _ru = shift_operator(mod_grad_u * grad_rho0, dx)
    rho_u = sum_over_dims(_ru) / rho0

  # Put everything together
  return nabla_u - rho_u, params

@operator
def wavevector(
  u: Field,
  medium: Medium,
  omega = 1.0,
  params = None
)  -> Field:
  r"""Wavevector operator for a generic `Field`.

  Args:
    u (Field): Complex field.
    medium (Medium): Medium object. Contains the value for `\alpha_0` and `c_0`.
    omega (float): Angular frequency.
    params (None, optional): Parameters for the operator.

  Returns:
    Field: Wavevector operator applied to `u`.
  """
  c = medium.sound_speed
  alpha = medium.attenuation
  trans_fun = lambda x: db2neper(x, 2.)
  alpha = compose(alpha)(trans_fun)
  k_mod = (omega / c) ** 2 + 2j * (omega ** 3) * alpha / c
  return u * k_mod, None

@operator
def helmholtz(
  u: Field,
  medium: Medium,
  omega = 1.0,
  params = None
) -> Field:
  r"""Evaluates the Helmholtz operator on a field $`u`$ with a PML.

  Args:
    u (Field): Complex field.
    medium (Medium): Medium object.
    omega (float): Angular frequency.
    params (None, optional): Parameters for the operator. **Unused**.

  Returns:
    Field: Helmholtz operator applied to `u`.
  """
  # Get the modified laplacian
  L = laplacian_with_pml(u, medium, omega)

  # Add the wavenumber term
  k = wavevector(u, medium, omega)
  return L + k, None


@operator
def helmholtz(
  u: OnGrid,
  medium: Medium,
  omega = 1.0,
  params = None
) -> OnGrid:
  r"""Evaluates the Helmholtz operator on a field $`u`$ with a PML. This
  implementation exposes the laplacian parameters to the user.

  Args:
    u (OnGrid): Complex field.
    medium (Medium): Medium object.
    omega (float): Angular frequency.
    params (None, optional): Parameters for the operator.

  Returns:
    OnGrid: Helmholtz operator applied to `u`.
  """
  if params == None:
    params = laplacian_with_pml.default_params(u, medium, omega)

  # Get the modified laplacian
  L = laplacian_with_pml(u, medium, omega, params=params)

  # Add the wavenumber term
  k = wavevector(u, medium, omega)
  return L + k, params


def scale_source_helmholtz(source, medium):
  if isinstance(medium.sound_speed, Field):
    min_sos = functional(medium.sound_speed)(jnp.amin)
  else:
    min_sos = jnp.amin(medium.sound_speed)

  source = source  * 2 / (source.domain.dx[0] * min_sos)
  return source
