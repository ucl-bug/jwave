from ._components import TimeHarmonicProblem
from ._solvers import HelmholtzSolver
from ._processors import HelmholtzProcessor, IdentityHelmholtzProcessor
from ._solution import Solution
from ._preconditioners import TimeHarmonicPreconditioner, IdentityPreconditioner

from jaxdf import Field
from typing import Union
from jax import jit

@jit
def solve_time_harmonic(
  problem: TimeHarmonicProblem,
  solver: HelmholtzSolver,
  source: Field,
  guess: Union[None, Field] = None,
  processor: HelmholtzProcessor = IdentityHelmholtzProcessor(),
  preconditioner: TimeHarmonicPreconditioner = IdentityPreconditioner(),
) -> Solution:
  # Pre-process the problem
  problem, source, guess, store = processor.preprocess(problem, source, guess)
  
  # Initialize the preconditioner
  preconditioner = preconditioner.initialize(problem, source, guess)
  
  # Solve the problem
  solution = solver(problem, source, guess, preconditioner)
  
  # Post-process the solution
  solution = processor.postprocess(solution, store)
  
  return solution
