from jaxdf import Module, Field
from jwave import Medium, Domain, FourierSeries
from jaxtyping import PyTree
from typing import Callable, Union
import abc
from jax import numpy as jnp
import equinox as eqx

# Generic interface for processors. They are responsible for 
# pre-processing the problem and post-processing the solution.
class HelmholtzProcessor(Module):
  @abc.abstractmethod
  def preprocess(
    self, 
    medium: Medium,
    source: Field,
    omega: float,
    guess: Union[None, Field] = None
  ):
    r"""The `preprocess` method should return the pre-processed 
    medium, source, omega and guess, the helmholtz operator to be used
    by the solver class and a `store` PyTree that will be used by
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
  def postprocess(self, field):
    raise NotImplementedError
  
  
# Example processor that does nothing
class NullHelmholtzProcessor(HelmholtzProcessor):
  def preprocess(self, medium, source, omega, guess=None):
    return medium, source, omega, guess, None
  
  def postprocess(self, field, store):
    return field
  
  
# The following scales the Helmholtz problem in natural units
class NormalizeHelmholtz(HelmholtzProcessor):
  
  def preprocess(self, medium, source, omega, guess=None):
    r"""Converts problem for the Convergent Born Series
    to work in natural units"""
    # Store conversion variables
    domain = medium.domain
    _store = {
        "dx": jnp.mean(jnp.asarray(domain.dx)),
        "omega": omega,
    }

    # Set discretization to 1
    dx = tuple(map(lambda x: x / _store["dx"], domain.dx))
    domain = Domain(domain.N, dx)

    # set omega to 1
    omega = 1.0

    # Update sound speed
    if issubclass(type(medium.sound_speed), FourierSeries):
        c = medium.sound_speed.params
    else:
        c = medium.sound_speed
    c = c / (_store["dx"] * _store["omega"])

    # Update fields
    source = FourierSeries(source.on_grid, domain)
    if issubclass(type(medium.sound_speed), FourierSeries):
        c = FourierSeries(c, domain)

    medium = medium.replace("sound_speed", c)

    return medium, source, omega, guess, _store

  def postprocess(self, field, store):
    domain = field.domain
    dx = tuple(map(lambda x: x * store["dx"], domain.dx))
    domain = Domain(domain.N, dx)

    return FourierSeries(field.params, domain)
  