from jaxdf import Module
from jwave import Domain, Field
from typing import Union
from .processors import HelmholtzProcessor, NullHelmholtzProcessor
from .preconditioners import TimeHarmonicPreconditioner, IdentityPreconditioner

import equinox as eqx


class TimeHarmonicProblemSettings(Module):
  pml_size: float = eqx.field(default=20.0, static=True)


class TimeHarmonicProblem(Module):
  domain: Domain
  frequency: float
  wavenumber: Field
  density: Field
  settings: TimeHarmonicProblemSettings = TimeHarmonicProblemSettings()
  

class HelmholtzSolver(Module):
  pass


def solve_time_harmonic(
  problem: TimeHarmonicProblem,
  solver: HelmholtzSolver,
  guess: Union[None, Field] = None,
  processor: HelmholtzProcessor = NullHelmholtzProcessor(),
  preconditioner: TimeHarmonicPreconditioner = IdentityPreconditioner(),
):
  # Pre-process the problem
  problem, guess, store = processor.preprocess(problem, guess)
  
  # Initialize the preconditioner
  preconditioner = preconditioner.initialize(problem)
  
  

class HelmholtzGMRES(HelmholtzSolver):
  