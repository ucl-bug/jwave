from jaxdf import Module, Field
import equinox as eqx
from ._components import TimeHarmonicProblem
from ._solution import Solution, IterativeTimeHarmonicSolution
from typing import Union, Tuple
from ._preconditioners import TimeHarmonicPreconditioner, IdentityPreconditioner
import abc
from jax import numpy as jnp
from jwave import scale_source_helmholtz, helmholtz
from jax import eval_shape, checkpoint
from jax.scipy.sparse.linalg import gmres
from typing import Callable
from jaxdf.operators import functional
from jax.lax import while_loop

class HelmholtzSolver(Module):
  @abc.abstractmethod
  def __call__(
    self,
    problem: TimeHarmonicProblem,
    source: Field,
    guess: Union[None, Field] = None,
    preconditioner: TimeHarmonicPreconditioner = IdentityPreconditioner()
  ) -> Solution:
    raise NotImplementedError
  
  def initialize_guess(
    self,
    problem: TimeHarmonicProblem,
    source: Field,
    preconditioner: TimeHarmonicPreconditioner
  ) -> Field:
    return source.replace_params(jnp.zeros_like(source.params))
  
class GMRESSolver(HelmholtzSolver):
  maxiter: int = 1000
  restart: int = 10
  rtol: float = 1e-3
  atol: float = 0.0
  solve_method: str = eqx.field(default="batched", static=True)
  checkpoint_operators: bool = eqx.field(default=False, static=True)

  def __check_init__(self):
    # Check that the solve method is correct
    assert self.solve_method in ["batched", "incremental"], f"Unknown solve method {self.solve_method} for GMRES. It should be either 'batched' or 'incremental'"
  
  def _apply_gmres(
    self,
    operator: Callable,
    source: Field,
    guess: Field,
    preconditioner: Callable,
  ) -> Tuple[Field, int]:
    return  gmres(
      operator,
      source,
      guess,
      M=preconditioner,
      tol=self.rtol,
      atol=self.atol,
      maxiter=self.maxiter,
      solve_method=self.solve_method,
    )
  

class HelmholtzGMRES(GMRESSolver):
  maxiter: int = 1000
  restart: int = 10
  rtol: float = 1e-3
  atol: float = 0.0
  solve_method: str = eqx.field(default="batched", static=True)
  checkpoint_operators: bool = eqx.field(default=False, static=True)
  
  def __call__(
    self,
    problem: TimeHarmonicProblem,
    source: Field,
    guess: Union[None, Field] = None,
    preconditioner: TimeHarmonicPreconditioner = IdentityPreconditioner()
  ) -> Solution:
    # Extract the medium and omega
    medium = problem.to_medium()
    omega = problem.omega
    
    # Scale the source field
    source = scale_source_helmholtz(source, medium)
    
    if self.checkpoint_operators:
      helm_func = checkpoint(helm_func)
      
    # Define guess if it is none
    if guess is None:
      guess = self.initialize_guess(problem, source, preconditioner)
    
    # Solve the system
    field, info = self._apply_gmres(
      operator=problem.helmholtz_operator,
      source=source,
      guess=guess,
      preconditioner=preconditioner.operator(medium, source)
    )
    
    # Rescale the field
    field = -1j * omega * field
    
    # Return the value of the solution
    return IterativeTimeHarmonicSolution(
      value = field, 
      converged = info
    )

def default_born_constants(problem: TimeHarmonicProblem):
  medium = problem.to_medium()
  omega = problem.omega
  
  k_max = omega / functional(medium.sound_speed)(jnp.amax).params
  k_min = omega / functional(medium.sound_speed)(jnp.amin).params
  k0 = jnp.sqrt(0.5 * (k_max**2 + k_min**2))
  
  epsilon = jnp.amax(jnp.abs((problem.k_sq.on_grid - k0**2)))
  
  return k0, epsilon

class BornSeries(HelmholtzSolver):
  get_constants: Callable = eqx.field(
    default=default_born_constants,
    static=True)
  maxiter: int = 1000
  tol: float = 1e-8
  
  def __call__(
    self,
    problem: TimeHarmonicProblem,
    source: Field,
    guess: Union[None, Field] = None,
    preconditioner: TimeHarmonicPreconditioner = IdentityPreconditioner()
  ) -> Solution:
    medium = problem.to_medium()
    omega = problem.omega
    preconditioner_operator = preconditioner.operator(medium, source)
    source = -source
    
    k0, epsilon = self.get_constants(problem)
    
    # Define guess if it is none
    if guess is None:
      guess = self.initialize_guess(problem, source, preconditioner)
      
    # Define resdidual operator
    def residual_operator(field):
      return problem.helmholtz_operator(field) + source
      
    # Setting up loop
    carry = (0, guess)
    norm_initial = jnp.linalg.norm(source.on_grid)
    
    def cond_fun(carry):
      numiter, field = carry
      cond_1 = numiter < self.maxiter
      cond_2 = jnp.linalg.norm(residual_operator(field).on_grid) / norm_initial > self.tol
      return cond_1 * cond_2
    
    def body_fun(carry):
      numiter, field = carry
      field = self.born_iteration(
        field, 
        problem, 
        source, 
        preconditioner_operator,
        k0=k0, 
        epsilon=epsilon)
      return numiter + 1, field
    
    numiters, out_field = while_loop(cond_fun, body_fun, carry)
    
    # Rescale field
    out_field = -1j * omega * out_field
    
    return IterativeTimeHarmonicSolution(value=out_field, converged=numiters)
  
  @staticmethod
  def born_iteration(
    field,
    problem,
    src,
    preconditioner,
    k0,
    epsilon
  ):
    V1 = problem.scattering_potential(field, k0=k0, epsilon=epsilon)
    G = problem.homogeneous_helmholtz_green(V1 + src, k0=k0, epsilon=epsilon)
    V2 = preconditioner(field - G)
    
    return field - V2
    