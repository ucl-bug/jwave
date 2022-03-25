from typing import Union

import jax
from jax import numpy as jnp
from jax.scipy.sparse.linalg import bicgstab, gmres
from jaxdf import operator
from jaxdf.discretization import OnGrid

from jwave.geometry import Medium

from .operators import helmholtz


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

  if params is None:
    params = helmholtz(source, medium, omega)._op_params

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
  return out, None


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

  return guess*src_magn
