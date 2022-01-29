from jwave.geometry import Medium
from jwave.acoustics.pml import complex_pml_on_grid
from typing import Union
from jaxdf.discretization import OnGrid
from jaxdf.operators import compose
from jaxdf import operator
from jax import numpy as jnp
from jax.scipy.sparse.linalg import bicgstab, gmres
import jax

from .operators import helmholtz

@operator
def helmholtz_solver(
  medium: Medium,
  omega: float,
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
    return gmres(helm_func, source, guess, tol=tol, restart=restart, maxiter=maxiter, solve_method=solve_method)
  elif method == 'bicgstab':
    return bicgstab(helm_func, source, guess, tol=tol, maxiter=maxiter)
    
def helmholtz_solver_verbose(
  medium: Medium,
  omega: float,
  source: OnGrid,
  guess: Union[OnGrid, None] = None,
  **kwargs
): 
  tol = kwargs['tol'] if 'tol' in kwargs else 1e-3
  residual_magnitude = jnp.linalg.norm(helmholtz(source, medium, omega).params)
  tol = tol * residual_magnitude
  maxiter = kwargs['maxiter'] if 'maxiter' in kwargs else 1000
  
  if guess is None:
    guess = source*0
  
  kwargs['maxiter'] = 1
  kwargs['tol'] = 0.0
  iterations = 0
  while residual_magnitude > tol and iterations < maxiter:
    guess = helmholtz_solver(medium, omega, source, guess, 'gmres', **kwargs)
    
    residual = helmholtz(guess, medium, omega) - source
    residual_magnitude = jnp.linalg.norm(residual.params)
    iterations += 1
    
    # Print iteration info
    print(
        f"Iteration {iterations}: residual magnitude = {residual_magnitude}, tol = {tol:.2e}",
        flush=True,
    )
  
  return guess
