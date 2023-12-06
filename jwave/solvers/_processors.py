import abc
from typing import Union
from jaxdf import Field, Domain, FourierSeries, Module
from ._components import TimeHarmonicProblem, TimeHarmonicProblemSettings
from typing import Tuple, Callable
from ._solution import Solution
from jaxtyping import PyTree
from jax import numpy as jnp
import numpy as np
from jwave import Medium
import equinox as eqx
from math import factorial

class HelmholtzProcessor(Module):
  @abc.abstractmethod
  def preprocess(
    self,
    problem: TimeHarmonicProblem,
    source: Field,
    guess: Union[None, Field] = None,
  ) -> Tuple[TimeHarmonicProblem, Field, Union[None, Field], PyTree]:
    r"""The `preprocess` method should return the pre-processed 
    problem and guess, and a `store` PyTree that will be used by
    the post-processing method.
    
    !!! "Signature
    ```python
    def preprocess(self, medium, source, omega, guess):
      ... # Do stuff
      return (medium, source, omega, guess, store)
    ```
    """
    raise NotImplementedError
  
  @abc.abstractmethod
  def postprocess(
    self, 
    solution: Solution,
    store: PyTree
  ) -> Solution:
    raise NotImplementedError

class IdentityHelmholtzProcessor(HelmholtzProcessor):
  def preprocess(
    self,
    problem: TimeHarmonicProblem,
    source: Field,
    guess: Union[None, Field] = None,
  ) -> Tuple[TimeHarmonicProblem, Field, Union[None, Field], PyTree]:
    return problem, source, guess, None
  
  def postprocess(
    self, 
    solution: Solution,
    store: PyTree
  ) -> Solution:
    return solution
  
class ChainPreprocessor(HelmholtzProcessor):
  processors: Tuple[HelmholtzProcessor]
  
  def preprocess(
    self,
    problem: TimeHarmonicProblem,
    source: Field,
    guess: Union[None, Field] = None,
  ) -> Tuple[TimeHarmonicProblem, Field, Union[None, Field], PyTree]:
    store = []
    for processor in self.processors:
      problem, source, guess, _store = processor.preprocess(problem, source, guess)
      store.append(_store)
    return problem, source, guess, store
  
  def postprocess(
    self, 
    solution: Solution,
    store: PyTree
  ) -> Solution:
    # Does post-processing in reverse order
    for processor, _store in zip(reversed(self.processors), reversed(store)):
      solution = processor.postprocess(solution, _store)
    
    return solution
  
class NormalizeHelmholtz(HelmholtzProcessor):
  r"""Normalizes the Helmholtz problem in natural units"""
  
  def preprocess(
    self,
    problem: TimeHarmonicProblem,
    source: Field,
    guess: Union[None, Field] = None,
  ) -> Tuple[TimeHarmonicProblem, Field, Union[None, Field], PyTree]:
    # Store conversion variables
    domain = problem.domain
    omega = problem.omega
    medium = problem.to_medium()
    
    _store = {
        "omega": omega,
    }
    
    # set omega to 1
    omega = 1.0
    
    # Update sound speed
    if issubclass(type(medium.sound_speed), FourierSeries):
        c = medium.sound_speed.params
    else:
        c = medium.sound_speed
    c = c / (_store["omega"])
    
    if issubclass(type(medium.sound_speed), FourierSeries):
        c = FourierSeries(c, domain)
    source = FourierSeries(source.on_grid, domain)
    if guess is not None:
      guess = FourierSeries(guess.on_grid, domain)
    
    medium = Medium(
      domain=domain,
      sound_speed=c,
      density=medium.density,
      attenuation=medium.attenuation,
      pml_size=medium.pml_size,
    )
    frequency = omega / (2 * jnp.pi)
    
    new_problem = TimeHarmonicProblem.from_medium(medium, frequency)
    return new_problem, source, guess, _store
  
  def postprocess(
    self,
    solution: Solution,
    store: PyTree
  ) -> Solution:
    return solution
  
class OsnabruggeBC(HelmholtzProcessor):
  pml_size: int = eqx.field(default=50, static=True)
  k0: Callable = eqx.field(
    default = lambda problem, source, guess: 1.0,
    static = True
  )
  alpha: float = eqx.field(default=1.0, static=True)
  N: int = eqx.field(default=4, static=True)
  
  def enlarge_domain(
    self,
    domain: Domain,
  ):
    new_N = tuple([x + 2 * self.pml_size for x in domain.N])
    return Domain(new_N, domain.dx)
  
  def pad_fun(
    self, 
    u: FourierSeries,
    new_domain: Domain,
    mode="constant"
  ):
    pad_size = tuple([(self.pml_size, self.pml_size)
                      for _ in range(len(new_domain.N))] + [(0, 0)])
    new_u = jnp.pad(u, pad_size, mode=mode)
    return new_u
  
  def make_pml(
    self,
    new_domain: Domain,
    k0: float
  ):
    def pml_edge(x):
        return x / 2 - self.pml_size
      
    def num(x):
        return (self.alpha**2) * (self.N - self.alpha * x + 2j * k0 * x) * (
            (self.alpha * x)**(self.N - 1))

    def den(x):
        return sum([((self.alpha * x)**i) / float(factorial(i))
                    for i in range(self.N + 1)]) * factorial(self.N)

    def transform_fun(x):
        return num(x) / den(x)
      
    # Make the PML field from a coordinate grid
    delta_pml = jnp.asarray(list(map(pml_edge, new_domain.N)))
    coord_grid = Domain(
      N=new_domain.N,
      dx=tuple([1.0] * len(new_domain.N))
    ).grid
    diff = jnp.abs(coord_grid) - delta_pml
    diff = jnp.where(diff > 0, diff, 0) / 4
    
    dist = jnp.sqrt(jnp.sum(diff**2, -1))
    k_k0 = transform_fun(dist)
    k_k0 = jnp.expand_dims(k_k0, -1)
    
    return k_k0
  
  def preprocess(
    self,
    problem: TimeHarmonicProblem,
    source: Field,
    guess: Union[None, Field],
  ) -> Tuple[TimeHarmonicProblem, Field, Union[None, Field], PyTree]:
    
    # Calculate reference k
    k0 = self.k0(problem, source, guess)
    
    # Extract fields on the grid
    _src = source.on_grid
    _k_sq = problem.k_sq.on_grid
    _density = problem.to_medium().density.on_grid
    _guess = guess.on_grid if guess is not None else None
    
    # Find new domain
    new_domain = self.enlarge_domain(problem.domain)

    # Pad fields
    _src = self.pad_fun(_src, new_domain)
    _k_sq = self.pad_fun(_k_sq, new_domain, mode="edge")
    _density = self.pad_fun(_density, new_domain, mode="edge")
    _guess = self.pad_fun(_guess, new_domain) if guess is not None else None

    # Make pml
    k_k0 = self.make_pml(new_domain, k0)
  
    # Update the problem k_sq 
    _k_sq = k_k0 + _k_sq
    
    # Build fields
    k_sq = FourierSeries(_k_sq, new_domain)
    source = FourierSeries(_src, new_domain)
    guess = FourierSeries(_guess, new_domain) if guess is not None else None
    density = FourierSeries(_density, new_domain)
    
    # Return the updated problem (it does not need a pml anymore)
    new_problem = TimeHarmonicProblem(
      domain = new_domain,
      frequency = problem.frequency,
      k_sq = k_sq,
      density = density,
      settings = TimeHarmonicProblemSettings(pml_size=0.0)
    )
    
    return new_problem, source, guess, None
  
  def postprocess(
    self,
    solution: Solution,
    store: PyTree
  ) -> Solution:
    # Remove the pml
    out_field = solution.value
    P = self.pml_size
    
    num_dims = len(out_field.domain.N)
    if num_dims == 1:
        _out_field = out_field.on_grid[P:-P]
    elif num_dims == 2:
        _out_field = out_field.on_grid[P:-P,P:-P]
    elif num_dims == 3:
        _out_field = out_field.on_grid[P:-P,P:-P,P:-P]
    else:
        raise ValueError("Only 1, 2, or 3 dimensions are supported.")

    # Build new field
    new_field = FourierSeries(_out_field, out_field.domain)
    
    solution = solution.replace("value", new_field)
    
    return solution