from jwave import Field
from jaxdf import Module
from equinox import Enumeration


# A custom solution class to work with diffrax, lineax, jaxdf and jwave
# Loosely based on https://docs.kidger.site/lineax/api/solution/
class Solution(Module):
  value: Field
  
  
# TODO : Add more details about the result  
class IterativeTimeHarmonicSolution(Solution):
  converged: int