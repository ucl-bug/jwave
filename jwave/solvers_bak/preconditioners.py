from jwave import Medium, Field
from jaxdf import Module
from typing import Union
from equinox import field
from equinox import Enumeration
from lineax import positive_semidefinite_tag 
import abc

class TimeHarmonicPreconditioner(Module):
  
  def initialize(self, medium: Medium, omega: float):
    r"""This should return the initialized pre conditioner. By
    default, it just returns itself."""
    return self
  
  @abc.abstractmethod
  def tags(self):
    r"""This should return a list of tags that describe the preconditioner.
    
    They should be one of the [`lineax tags`](https://docs.kidger.site/lineax/api/tags/)
    """
    raise NotImplementedError
  
  @abc.abstractmethod
  def __call__(
    self, 
    medium: Medium, 
    source: Field, 
    field: Field
  ) -> Field:
    r"""This must be a linear function of the field."""
    raise NotImplementedError(
      f"Preconditioner {self.__class__.__name__} call is not implemented"
    )
    
  def _operator(
    self,
    medium,
    source
  ):
    return lambda field: self(medium, source, field)


class IdentityPreconditioner(TimeHarmonicPreconditioner):
  # This preconditioner does nothing
  def __call__(self, medium: Medium, source: Field, field: Field) -> Field:
    return field
  
  @property
  def tags(self):
    return positive_semidefinite_tag
