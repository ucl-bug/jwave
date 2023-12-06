from math import factorial
from jaxdf import Module
from .preconditioners import TimeHarmonicPreconditioner, IdentityPreconditioner
from .solution import Solution
from jwave import Medium, Field, OnGrid
from jaxdf import Module
from typing import Union, Callable, Tuple
from jwave import scale_source_helmholtz, helmholtz
from jax import eval_shape, checkpoint
from jax.scipy.sparse.linalg import gmres
import equinox as eqx
from jax import numpy as jnp
from jwave import FourierSeries, Domain
from jaxdf import laplacian, functional
from jwave import wavevector
from jwave import db2neper, neper2db


class HelmholtzSolver(Module):
  pass 

class HelmholtzGMRES(HelmholtzSolver):
  maxiter: int = 1000
  restart: int = 10
  rtol: float = 1e-3
  atol: float = 0.0
  solve_method: str = eqx.field(default="batched", static=True)
  checkpoint_operators: bool = eqx.field(default=False, static=True)
  
  def __call__(
    self,
    medium: Medium,
    source: OnGrid,
    omega: float,
    guess: Union[OnGrid, None] = None,
    preconditioner: TimeHarmonicPreconditioner = IdentityPreconditioner()
  ):
    # Initialize preconditioner
    preconditioner = preconditioner.initialize(medium, omega)
    
    # Scale the source field
    source = scale_source_helmholtz(source, medium)

    # Initialize guess
    if guess is None:
      guess = source * 0j

    # Define the helmholtz linear operator
    def helm_func(field):
      return helmholtz(field, medium, omega=omega)

    if self.checkpoint_operators:
      helm_func = checkpoint(helm_func)
    
    # Define the preconditioning operator
    def precond(field):
      return preconditioner(medium, source, field)
    
    if self.checkpoint_operators:
      precond = checkpoint(precond)
      
    # Check that the solve method is correct
    assert self.solve_method in ["batched", "incremental"], f"Unknown solve method {self.solve_method} for GMRES. It should be either 'batched' or 'incremental'"

    # Solve the system
    field, info = gmres(
      helm_func,
      source,
      guess,
      M=precond,
      tol=self.rtol,
      atol=self.atol,
      maxiter=self.maxiter,
      solve_method=self.solve_method,
    )
    
    # Rescale the field
    field = -1j * omega * field
    
    # Return the value of the solution
    return Solution(value = field, converged = info)

def default_born_k0_init(k_sq):
  k_max = jnp.amax(k_sq.on_grid)
  k_min = jnp.amin(k_sq.on_grid)
  k0 = jnp.sqrt(0.5 * (k_max + k_min))
  return k0

class BornSeries(HelmholtzSolver):
  k0_init: Callable = eqx.field(default = default_born_k0_init, static = True)
  epsilon_init: Callable = eqx.field(
    default =  lambda k_sq, k0_sq: jnp.amax(jnp.abs((k_sq.on_grid - k0_sq**2))),
    static = True
  )
  k_pml: Callable = eqx.field(
    default = lambda m, omega: jnp.amax(omega / m.sound_speed.on_grid),
    static = True
  )
  maxiter: int = 1000
  atol: float = 1e-8
  alpha: float = 1.0
  
  @staticmethod
  def enlarge_domain(domain, pml_size):
    new_N = tuple([x + 2 * pml_size for x in domain.N])
    return Domain(new_N, domain.dx)
  
  def pad_fun(
    self, 
    u, 
    pml_size,
    **kwargs
  ):
    pad_size = tuple([(pml_size, pml_size)
                      for _ in range(len(u.domain.N))] + [(0, 0)])
    return FourierSeries(jnp.pad(
      u.on_grid, pad_size, **kwargs),
      self.enlarge_domain(u.domain, pml_size)
    )
    
  @staticmethod
  def helmholtz_operator(field, k_sq):
    return laplacian(field) + k_sq * field
  
  def k_sq_pml(
    self, 
    N: tuple[int, int],
    pml_size,
    k_pml: float,
    alpha: float
  ):
    N = 4

    def pml_edge(x):
        return x / 2 - pml_size

    def num(x):
        return (alpha**2) * (N - alpha * x + 2j * k_pml * x) * (
            (alpha * x)**(N - 1))

    def den(x):
        return sum([((alpha * x)**i) / float(factorial(i))
                    for i in range(N + 1)]) * factorial(N)

    def transform_fun(x):
        return num(x) / den(x)

    delta_pml = jnp.asarray(list(map(pml_edge, N)))
    coord_grid = Domain(N=N, dx=tuple([1.0] * len(N))).grid
    coord_grid = coord_grid

    diff = jnp.abs(coord_grid) - delta_pml
    diff = jnp.where(diff > 0, diff, 0) / 4

    dist = jnp.sqrt(jnp.sum(diff**2, -1))
    k_k_pml = transform_fun(dist)
    k_k_pml = jnp.expand_dims(k_k_pml, -1)
    return k_k_pml + k_pml**2
  
  @staticmethod
  def remove_pml(field, pml_size):
    num_dims = len(field.domain.N)
    
    if num_dims == 1:
        _field = field.on_grid[pml_size:-pml_size]
    elif num_dims == 2:
        _field = field.on_grid[pml_size:-pml_size,
                                      pml_size:-pml_size]
    elif num_dims == 3:
        _field = field.on_grid[pml_size:-pml_size,
                                      pml_size:-pml_size,
                                      pml_size:-pml_size]
    
    return FourierSeries(_field, field.domain)
  
  def construct_problem(
    self,
    medium: Medium,
    source: FourierSeries,
    omega: float,
    guess: Union[FourierSeries, None] = None,
  ):  
    # Get constants
    pml_size = medium.pml_size
      
    # Generate guess if not defined
    if guess is None:
      guess = source * 0j
      
    # Scale the source
    source = scale_source_helmholtz(source, medium)
      
    # Pad the fields
    source = self.pad_fun(source, medium.pml_size)
    guess = self.pad_fun(guess, medium.pml_size)
    
    # Construct the heterogeneous k_sq
    ones_field = FourierSeries(jnp.ones(source.domain.N), source.domain)
    k_original = wavevector(
      ones_field, medium, omega=omega
    ).on_grid[...,0]
    
    k_pml = self.k_pml(medium, omega)
    k_sq_pml = self.k_sq_pml(
      source.domain.N,
      pml_size,
      k_pml,
      self.alpha
    )
    
    PS = pml_size
    k_sq = k_sq_pml.at[PS:-PS, PS:-PS].set(k_original)
    del PS
    
    # Transform them into medium
    sound_speed = jnp.sqrt(jnp.real(k_sq) / omega)
    alpha_neper = jnp.imag(k_sq) * sound_speed / (omega**2)
    alpha = neper2db(alpha_neper)
    sound_speed = FourierSeries(sound_speed, source.domain)
    alpha = FourierSeries(alpha, source.domain)
    
    medium = Medium(
      domain = source.domain,
      sound_speed = sound_speed,
      attenuation = alpha,
      pml_size = pml_size
    )
    
    return medium, source, omega, guess
    
  def born_loop(
    self,
    k_sq,
    k0,
    epsilon,
    source,
    guess,
    preconditioner
  ):
    
    carry = (0, guess, k0)  
    
    # Get the source norm
    source_norm = jnp.linalg.norm(source.on_grid)
    
    def resid_fun(field, k_sq, src):
      return laplacian(field) + k_sq * field + src
    
    def cond_fun(carry):
      numiter, field = carry
      cond_1 = numiter < self.maxiter
      cond_2 = jnp.linalg.norm(resid_fun(field, k_sq, source).on_grid) / source_norm > self.atol
      return cond_1 & cond_2
    
    def body_fun(carry):
      numiter, field = carry
      field = self.born_iteration(
        field, k_sq, source, k0=k0, epsilon = epsilon
      )
      return (numiter + 1, field), field

  def born_iteration(
    self,
  ):
    
  
  def __call__(
    self,
    medium: Medium,
    source: OnGrid,
    omega: float,
    guess: Union[OnGrid, None] = None,
    preconditioner: TimeHarmonicPreconditioner = IdentityPreconditioner()
  ):
    # Construct the problem
    medium, source, guess, k_sq = self.construct_problem(
      medium, source, omega, guess, preconditioner
    )
    
    # Initialize preconditioner
    preconditioner = preconditioner.initialize(medium, omega)
    
    # Find k0
    k0 = self.k0_init(k_sq)
    
    # Find epsilon
    epsilon = self.epsilon_init(k_sq, k0)
    
    