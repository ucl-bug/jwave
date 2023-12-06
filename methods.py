from jwave import Medium, Source, HelmholtzSolver, Field
from jaxdf import Module
from typing import Union

# Operators

# Preconditioners
class TimeHarmonicPreconditioner(Module):
  initialized: bool
  
  def initialize(self, medium: Medium, omega: float,):
    pass
  
  def __call__(self, medium: Medium, source: Source, field: Field) -> Field:
    raise NotImplementedError(f"Preconditioner {self.__class__.__name__} call is not implemented")


class IdentityPreconditioner(Module):
  # This preconditioner does nothing
  def __call__(self, medium: Medium, source: Source, field: Field) -> Field:
    return field


class OsnabruggePreconditioner(TimeHarmonicPreconditioner):
  k0: float
  eps: float
  
  
# PML Constructor
class HelmholtzPreProcessor(Module):
  pass

class PMLConstructor(Module):
  pass
  
# Solvers
class HelmholtzSolver(Module):
  pass 


class HelmholtzGMRES(HelmholtzSolver):
  restart: int
  maxiter: int
  tol: float
  atol: float
  preconditioner: Module = IdentityPreconditioner()
  
class LippmanSchwingerGMRES(HelmholtzSolver):
  restart: int
  maxiter: int
  tol: float
  atol: float
  preconditioner: Module = IdentityPreconditioner()
  
class HomotopySeries(HelmholtzSolver):
  k0: float
  eps: float
  alpha: float
  maxiter: int
  tol: float
  atol: float
  preconditioner: Module = IdentityPreconditioner()
  
class ConvergentBornSeries(HelmholtzSolver):
  k0: float
  eps: float
  alpha: float
  
class NeuralNetwork(HelmholtzSolver):
  neural_network: Module
  

# Final solver
class TimeHarmonicSolver(Module):
  solver: HelmholtzSolver
  processor: Union[None, HelmholtzPreProcessor] = None
  
  def __call__(
    self, medium: Medium, 
    source: Source, 
    omega: float,
    *, 
    guess: Union[None, Field] = None
  ) -> Field:
    # Construct problem
    medium, source, guess, helmholtz_operator, _store = self.processor.preprocess(
      medium, source, omega
    )
    
    # Solve with the given solver
    field = self.solver(helmholtz_operator, medium, source, guess=guess)
    
    # Postprocess
    field = self.processor.postprocess(field, _store)
    
####################### Examples #######################
medium = ...
source = ...
omega = ...
guess = ...
k0 = ...
eps = ...
alpha = ...

# simple gmres
simple_gmres = TimeHarmonicSolver(
  processor = ScaleHelmholtz,
  solver = HelmholtzGMRES  
)
sol = simple_gmres(medium, source, omega, guess=guess)

# simple LS gmres
simple_ls_gmres = TimeHarmonicSolver(
  processor = ScaleHelmholtz,
  solver = LippmanSchwingerGMRES  
)
sol = simple_ls_gmres(medium, source, omega, guess=guess)

# Born series
born_series = TimeHarmonicSolver(
  processor = ChainPreprocessor(
    ScaleHelmholtz(),
    OsnabruggeABC(k0),
  ),
  solver = HomotopySeries(k0, eps, alpha)
)
sol = born_series(medium, source, omega, guess=guess)

# Convergent Born series
convergent_born_series = TimeHarmonicSolver(
  processor = Chainprocessoror(
    ScaleHelmholtz(),
    OsnabruggeABC(k0),
  ),
  solver = HomotopySeries(k0, eps, alpha)
)
sol = convergent_born_series(
  medium, source, omega,
  guess=guess,
  preconditioner=OsnabruggePreconditioner(k0, eps)
  )

# Reuse preconditioner
osnabrugge = OsnabruggePreconditioner(k0, eps).initialize(medium, omega)
convergent_born_series = TimeHarmonicSolver(
  processor = ChainPreprocessor(
    ScaleHelmholtz(),
    OsnabruggeABC(k0),
  ),
  solver = HomotopySeries(k0, eps, alpha)
)
sol = convergent_born_series(
  medium, source, omega,
  guess=guess,
  preconditioner=osnabrugge
)

# Low-rank preconditioner
born_low_rank = TimeHarmonicSolver(
  processor = ChainPreprocessor(
    ScaleHelmholtz(),
    OsnabruggeABC(k0),
  ),
  solver = HomotopySeries(k0, eps, alpha)
)
sol = born_low_rank(
  medium, source, omega,
  guess=guess,
  preconditioner=LowRankFFTPreconditioner(rank=10)
)

# Hierarchical preconditioner
born_hierarchical = TimeHarmonicSolver(
  processor = ChainPreprocessor(
    ScaleHelmholtz(),
    OsnabruggeABC(k0),
  ),
  solver = HomotopySeries(k0, eps, alpha)
)
sol = born_hierarchical(
  medium, source, omega,
  guess=guess,
  preconditioner=HierarchicalFFTPreconditioner(depth=3, rank=10)
)

# Gradient descent solver
gd_solver = TimeHarmonicSolver(
  processor = ChainPreprocessor(
    ScaleHelmholtz(),
    OsnabruggeABC(k0),
  ),
  solver = GradientDescentSolver(
    optimizer = adam(1e-3),
    maxiter = 1000,
    tol = 1e-6,
  )
)
field = ... # Neural network representation
sol = gd_solver(medium, source, omega, guess=field)