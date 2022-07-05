from typing import Union

import jax
from jax import numpy as jnp
from jax.scipy.sparse.linalg import bicgstab, gmres
from jaxdf import operator
from jaxdf.discretization import Field, FourierSeries, OnGrid
from jaxdf.geometry import Domain
from jaxdf.operators import functional

from jwave.geometry import Medium

from .conversion import db2neper
from .operators import helmholtz


@operator
def angular_spectrum(
  pressure: FourierSeries,
  *,
  z_pos: float,
  f0: float,
  medium: Medium,
  padding: int = 0,
  angular_restriction: bool = True,
  params = None
) -> FourierSeries:
  """Similar to `angularSpectrumCW` from the k-Wave toolbox.

  Projects an input plane of single-frequency
  continuous wave data to the parallel plane specified by z_pos using the
  angular spectrum method. The implementation follows the spectral
  propagator with angular restriction described in reference [1].

  For linear projections in a lossless medium, just the sound speed can
  be specified. For projections in a lossy medium, the parameters are
  given as fields to the input structure medium.

  See [[Zeng and McGhough, 2008](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3408224/)] for
  more details.


  Args:
      pressure (FourierSeries): omplex pressure values over the input plane $`[Pa]`$
      z_pos (float): Specifies the relative z-position of the plane of projection.
      f0 (float): The frequency of the input plane.
      medium (Medium): Specifies the speed of sound, density and absorption in the medium.
      padding (Union[str, int], optional): Controls the grid expansion used for
        evaluating the angular spectrum. Defaults to 0.
      angular_restriction (bool, optional): If true, uses the angular restriction method
        specified in [[Zeng and McGhough, 2008](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3408224/)]. Defaults to True.

  Returns:
      FourierSeries: _description_
  """
  # Get literals
  c0 = medium.sound_speed
  k = 2 * jnp.pi * f0 / c0
  k_t_sq = k**2

  # Pad the input field
  p = pressure.on_grid
  p = jnp.pad(p, padding, mode='constant', constant_values=0)

  # Update the domain
  domain = Domain(
    (p.shape[0], p.shape[1]), # (Nx, Ny)
    pressure.domain.dx        # Grid spacing doesn't change
  )
  pressure_padded = FourierSeries(p, domain)

  # Define cutoffs
  freq_grid = pressure_padded._freq_grid
  k_x_sq = jnp.sum(freq_grid ** 2, axis=-1)
  kz = jnp.sqrt(k_t_sq - k_x_sq +0j)

  # Evaluate base propagator
  H = jnp.conj(jnp.exp(1j * z_pos * kz))

  # Apply angular restriction, i.e. a hard low-pass filter
  D = min(pressure_padded.domain.size)
  kc = k * jnp.sqrt(0.5 * D**2 / (0.5 * D**2 + z_pos**2))
  H_restrict = jnp.where(k_x_sq <= kc**2, H, 0.j)
  H = jnp.where(angular_restriction, H_restrict, H)

  # Add attenuation
  alpha_np = db2neper(medium.attenuation, 2)
  H = H = H * jnp.exp(-alpha_np * z_pos * k / kz)

  # Apply the spectral porpagator
  p_hat = jnp.fft.fftn(pressure_padded.on_grid[...,0])
  p_hat_plane = p_hat * H
  p_plane = jnp.fft.ifftn(p_hat_plane)

  # Unpad
  if padding > 0:
    p_plane = p_plane[padding:-padding, padding:-padding]

  return FourierSeries(p_plane, domain)


@operator
def rayleigh_integral(
  pressure: FourierSeries,
  r: jnp.ndarray,
  f0: float,
  sound_speed: float = 1500.0,
  params=None,
):
  """
  Rayleigh integral for a `FourierSeries` field.

  Args:
    pressure (FourierSeries): pressure field, corresponding to $`u`$ on the plane.
    r (jnp.ndarray): distance from the origin of the pressure plane.
      Must be a 3D array.
    f0 (float): frequency of the source.
    sound_speed (float): Value of the homogeneous sound speed where
      the rayleigh integral is computed. Default is 1500 m/s.

  Returns:
    complex64: Rayleigh integral at `r`
  """
  # TODO: Override vmap on second dimension using Fourier implementation of the rayleigh integral
  #       instead of the direct implementation. See Appendix B of
  #       https://asa.scitation.org/doi/pdf/10.1121/1.4928396 for details.

  # Checks
  if pressure.ndim != 2:
    raise ValueError("Only 2D domains are supported.")

  assert r.shape == (3,), "The target position must be a 3D vector."

  # Terms in the Rayleigh integral
  # See eq. A2 and A3 in https://asa.scitation.org/doi/10.1121/1.4928396
  k = 2*jnp.pi*f0/sound_speed
  def exp_term(x, y ,z):
    """The exponential term of the Rayleigh integral
    (first kind). This is basically the Green's function
    of a dirac delta with Sommerfield radiation conditions."""
    r = jnp.sqrt(x**2 + y**2 + z**2)
    return jnp.exp(1j*k*r)/r

  def direc_exp_term(x,y,z):
    """Derivative of the exponential term in the Rayleigh integral,
    along the z-axis (second kind). This is basically the Green's function
    of a dipole oriented along the z-axis.
    """
    _, direc_derivative = jax.jvp(exp_term, (x,y,z), (0., 0., 1.))
    return direc_derivative

  # Integral calculation as a finite sum
  area = pressure.domain.cell_volume
  plane_grid = pressure.domain.grid

  # Append z dimension of zeros to the last dimension
  z_dim = jnp.zeros(plane_grid.shape[:-1] + (1,))
  plane_grid = jnp.concatenate((plane_grid, z_dim), axis=-1)

  # Distance from r to the plane
  R = jnp.abs(r - plane_grid)

  # Weights of the Rayleigh integral
  weights = jax.vmap(jax.vmap(direc_exp_term, in_axes=(0,0,0)), in_axes=(0,0,0))(
      R[...,0], R[...,1], R[...,2]
  )
  return jnp.sum(weights*pressure.on_grid)*area, None

@operator
def helmholtz_solver(
  medium: Medium,
  omega: object,
  source: OnGrid,
  guess: Union[OnGrid, None] = None,
  method: str = 'gmres',
  checkpoint: bool = True,
  params = None,
  **kwargs
):
  """_summary_

  Args:
      medium (Medium): _description_
      omega (object): _description_
      source (OnGrid): _description_
      guess (Union[OnGrid, None], optional): _description_. Defaults to None.
      method (str, optional): _description_. Defaults to 'gmres'.
      checkpoint (bool, optional): _description_. Defaults to True.
      params (_type_, optional): _description_. Defaults to None.

  Returns:
      _type_: _description_
  """
  if isinstance(medium.sound_speed, Field):
    min_sos = functional(medium.sound_speed)(jnp.amin)
  else:
    min_sos = jnp.amin(medium.sound_speed)

  source = source  * 2 / (source.domain.dx[0] * min_sos)

  if params is None:
    params = helmholtz.default_params(source, medium, omega)

  def helm_func(u):
    return helmholtz(u, medium, omega, params=params)

  if checkpoint:
    helm_func = jax.checkpoint(helm_func)

  if guess is None:
    guess = source*0

  tol = kwargs['tol'] if 'tol' in kwargs else 1e-3
  restart = kwargs['restart'] if 'restart' in kwargs else 10
  maxiter = kwargs['maxiter'] if 'maxiter' in kwargs else 1000
  solve_method = kwargs['solve_method'] if 'solve_method' in kwargs else 'batched'
  if method == 'gmres':
    out = gmres(helm_func, source, guess, tol=tol, restart=restart, maxiter=maxiter, solve_method=solve_method)[0]
  elif method == 'bicgstab':
    out = bicgstab(helm_func, source, guess, tol=tol, maxiter=maxiter)[0]
  return -1j*omega*out, None


def helmholtz_solver_verbose(
  medium: Medium,
  omega: float,
  source: OnGrid,
  guess: Union[OnGrid, None] = None,
  params=None,
  **kwargs
):

  src_magn = jnp.linalg.norm(source.on_grid)
  source = source / src_magn

  tol = kwargs['tol'] if 'tol' in kwargs else 1e-3
  residual_magnitude = jnp.linalg.norm(helmholtz(source, medium, omega).params)
  maxiter = kwargs['maxiter'] if 'maxiter' in kwargs else 1000

  if params is None:
    params = helmholtz(source, medium, omega)._op_params

  if guess is None:
    guess = source*0

  kwargs['maxiter'] = 1
  kwargs['tol'] = 0.0
  iterations = 0

  if isinstance(medium.sound_speed, Field):
    min_sos = functional(medium.sound_speed)(jnp.amin)
  else:
    min_sos = jnp.amin(medium.sound_speed)

  @jax.jit
  def solver(medium, guess, source):
    guess = helmholtz_solver(medium, omega, source, guess, 'gmres', **kwargs, params=params)
    residual = helmholtz(guess, medium, omega, params=params) - source
    residual_magnitude = jnp.linalg.norm(residual.params)
    return guess, residual_magnitude

  while residual_magnitude > tol and iterations < maxiter:
    guess, residual_magnitude = solver(medium, guess, source)

    iterations += 1

    # Print iteration info
    print(
        f"Iteration {iterations}: residual magnitude = {residual_magnitude:.4e}, tol = {tol:.2e}",
        flush=True,
    )

  return -1j*omega*guess* src_magn * 2 / (source.domain.dx[0] * min_sos)
