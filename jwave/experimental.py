from functools import partial
from math import factorial

from jax import numpy as jnp
from jax.lax import while_loop
from jaxdf import Field
from jaxdf.discretization import Field, FourierSeries, OnGrid

from jwave.acoustics.pml import _base_pml
from jwave.geometry import Domain, Medium


# Building base PML
def cbs_pml(
  field: OnGrid,
  k0: object = 1.0,
  pml_size: object = 32,
  alpha: object = 1.0
):
  medium = Medium(domain = field.domain, pml_size=pml_size)
  N = 4

  def num(x):
    return (alpha**2)*(N - alpha*x + 2j*k0*x)*((alpha*x)**(N-1))

  def den(x):
    return sum([((alpha*x)**i)/float(factorial(i)) for i in range(N)])*factorial(N)

  def transform_fun(x):
    x = x
    return num(x)/den(x)

  k_k0 = _base_pml(transform_fun, medium, exponent=1.0, alpha_max=2.)
  k_k0 = jnp.expand_dims(jnp.sum(k_k0, -1), -1)
  return k_k0 + k0**2

def cbs_helmholtz(
  field: OnGrid,
  k_sq: OnGrid
):
  laplacian = lapl_func(field)
  return laplacian + k_sq*field

def lapl_func(field):
  freq_grid = field._freq_grid
  p_sq = jnp.sum(freq_grid**2, -1)

  u = field.on_grid[...,0]
  u_fft = jnp.fft.fftn(u)
  Gu_fft = (-p_sq)* u_fft
  Gu = jnp.fft.ifftn(Gu_fft)
  return field.replace_params(jnp.expand_dims(Gu, -1))

def born_solver(
  sound_speed: FourierSeries,
  src: FourierSeries,
  omega: object = 1.0,
  k0: object = 1.0,
  pml_size: object = 32,
  max_iter: object = 1000,
  tol: object = 1e-6,
  alpha: object = 1.0
) -> FourierSeries:
  r"""Solves the Helmholtz equation (without absorption atm) using the
  Convergente Born Series method described in:

  Gerwin Osnabrugge, Saroch Leedumrongwatthanakun, Ivo M. Vellekoop,
  A convergent Born series for solving the inhomogeneous Helmholtz equation in arbitrarily large media, Journal of Computational Physics, https://doi.org/10.1016/j.jcp.2016.06.034.

  Args:
    sound_speed (FourierSeries): The sound speed field.
    src (FourierSeries): The complex source field.
    omega (object): The angular frequency.
    k0 (object): The wavenumber.
    pml_size (object): The size of the PML.
    max_iter (object): The maximum number of iterations.
    tol (object): The relative tolerance for the convergence.
    alpha (object): The amplitude parameter of the PML.

  Returns:
    FourierSeries: The complex solution field.
  """

  def pad_fun(u):
    new_N = tuple([x+(pml_size*2*src.domain.dx[0]) for x in src.domain.N])
    return FourierSeries(
      jnp.pad(u.on_grid, ((pml_size,pml_size), (pml_size,pml_size), (0,0))),
      Domain(new_N, src.domain.dx)
    )

  src = pad_fun(src)

  k_sq = cbs_pml(src, k0, pml_size, alpha)
  k_sq = k_sq.at[pml_size:-pml_size,pml_size:-pml_size].set(
    ((omega/sound_speed)**2).on_grid + 0j
  )
  k_sq = FourierSeries(k_sq, src.domain)
  plt.imshow(jnp.abs(k_sq.on_grid))
  plt.colorbar()
  plt.savefig('k_sq.png')
  plt.close()

  epsilon = jnp.amax(jnp.abs((k_sq.on_grid - k0**2)))
  print(epsilon)

  norm_initial = jnp.linalg.norm(src.on_grid)

  guess = jnp.zeros((src.domain.N[0], src.domain.N[1], 1)) + 0j
  carry = (0, guess)

  def resid_fun(field):
    return cbs_helmholtz(field, k_sq) + src

  def cond_fun(carry):
    numiter, field = carry
    field = FourierSeries(field, src.domain)
    cond_1 = numiter < max_iter
    cond_2 = jnp.linalg.norm(resid_fun(field).on_grid) / norm_initial > tol
    return cond_1*cond_2

  def body_fun(carry):
    numiter, field = carry
    field = FourierSeries(field, src.domain)
    field = born_iteration(field, k_sq, src, k0, epsilon)
    return numiter + 1, field.on_grid

  _, out_field = while_loop(cond_fun, body_fun, carry)

  # unpad
  out_field = out_field # [pml_size:-pml_size,pml_size:-pml_size,:]
  return jnp.sum(out_field, axis=-1), k_sq.on_grid

def born_iteration(
  field,
  k_sq,
  src,
  k0,
  epsilon
):
  _V = partial(V, k_sq=k_sq, k0=k0, epsilon=epsilon)
  _G = partial(homog_greens, k0=k0, epsilon=epsilon)

  return field - (1j/epsilon)*_V(field - _G(_V(field) + src))

def M(
  field: Field,
  k_sq: Field,
  k0: object = 1.0,
  epsilon: object = 0.1
) -> Field:
  _gamma = partial(gamma, k_sq=k_sq, k0=k0, epsilon=epsilon)
  _V = partial(V, k_sq=k_sq, k0=k0, epsilon=epsilon)
  _G = partial(homog_greens, k0=k0, epsilon=epsilon)
  return _gamma(_G(_V(field))) - _gamma(field) + field


def gamma(
  field: Field,
  k_sq: Field,
  k0: object = 1.0,
  epsilon: object = 0.1
) -> Field:
  return (1j/epsilon)*V(field, k_sq, k0, epsilon)

def V(
  field: Field,
  k_sq: Field,
  k0: object = 1.0,
  epsilon: object = 0.1
) -> Field:
  k = (k_sq -  k0**2 - 1j*epsilon)
  out = field * k
  return out

def homog_greens(
  field: FourierSeries,
  k0: object = 1.0,
  epsilon: object = 0.1
):
  freq_grid = field._freq_grid
  p_sq = jnp.sum(freq_grid**2, -1)

  g_fourier = 1.0 / (p_sq - (k0**2) - 1j*epsilon)
  u = field.on_grid[...,0]
  u_fft = jnp.fft.fftn(u)
  Gu_fft = g_fourier * u_fft
  Gu = jnp.fft.ifftn(Gu_fft)
  return field.replace_params(jnp.expand_dims(Gu, -1))

if __name__ == '__main__':
  from matplotlib import pyplot as plt

  domain = Domain((128,128),(1,1))
  sos = jnp.ones(domain.N)
  sos = sos.at[10:48,10:64].set(2.0)
  src = jnp.zeros_like(sos) + 0j
  src = src.at[96,32:96].set(jnp.exp(-1j*jnp.linspace(0,4,96-32)**2))

  src = FourierSeries(jnp.expand_dims(src, -1), domain)
  sos = FourierSeries(jnp.expand_dims(sos, -1), domain)

  solution, k_sq = born_solver(
    sos,
    src,
    omega=1.0,
    k0=1.0,
    pml_size=128,
    max_iter=1000,
    tol=1e-6,
    alpha=1.0
  )

  # Plot and save solution
  outfield = solution#k_sq[...,0]#
  outfield = outfield[128:-128,128:-128]
  maxval = jnp.max(jnp.abs(outfield))
  #plt.imshow(jnp.real(outfield), vmax=maxval, vmin=-maxval, cmap="seismic")
  plt.imshow(jnp.abs(outfield), vmax=maxval, cmap="inferno")
  plt.colorbar()
  plt.savefig('born_solution_test.png')
