from jaxdf import Module
from jwave import Domain, Field, FourierSeries
from typing import Union
from jax import numpy as jnp
from jwave.geometry import Medium
from jaxdf.operators import functional

from jwave import helmholtz

import equinox as eqx


class TimeHarmonicProblemSettings(Module):
  pml_size: float = eqx.field(default=20.0, static=True)


class TimeHarmonicProblem(Module):
  domain: Domain
  frequency: float
  k_sq: Field
  density: Field
  settings: TimeHarmonicProblemSettings = TimeHarmonicProblemSettings()
  
  @property
  def omega(self) -> float:
    return 2 * jnp.pi * self.frequency
  
  def to_medium(self) -> Medium:
    omega = self.omega
    omega_sq = omega ** 2
    
    sound_speed_sq = omega_sq / functional(self.k_sq)(jnp.real)
    sound_speed = functional(sound_speed_sq)(jnp.sqrt)
    
    attenuation = functional(self.k_sq)(jnp.imag) * sound_speed / (2 * omega**3)
    
    return Medium(
      domain=self.domain,
      density=self.density,
      sound_speed=sound_speed,
      attenuation=attenuation,
      pml_size=self.settings.pml_size,
    )
  
  @classmethod
  def from_medium(
    cls,
    medium: Medium,
    frequency: float,
  ) -> 'TimeHarmonicProblem':
    omega = 2 * jnp.pi * frequency
    omega_sq = omega ** 2
    
    real_part = omega_sq / (medium.sound_speed ** 2)
    imag_part = 2j * (omega**3) * medium.attenuation / medium.sound_speed
  
    k_sq = real_part + imag_part
    
    return cls(
      domain=medium.domain,
      frequency=frequency,
      k_sq=k_sq,
      density=medium.density,
      settings=TimeHarmonicProblemSettings(pml_size=medium.pml_size),
    )
  
  @property
  def helmholtz_operator(self):
    medium = self.to_medium()
    omega = self.omega
    
    def helmholtz_operator(field):
      return helmholtz(field, medium, omega=omega)
    
    return helmholtz_operator
  
  @staticmethod
  def homogeneous_helmholtz_green(
    field: FourierSeries,
    k0=1.0,
    epsilon=0.1
  ):
    r"""Implements the Green's operator for the homogeneous Helmholtz equation.

    Note that being the field a `FourierSeries`, the Green's function is periodic.

    Args:
      field (FourierSeries): The input field $u$.
      k0 (object): The wavenumber.
      epsilon (object): The absorption parameter.

    Returns:
      FourierSeries: The result of the Green's operator on $u$.
    """
    freq_grid = field._freq_grid
    p_sq = jnp.sum(freq_grid**2, -1)

    g_fourier = 1.0 / (p_sq - (k0**2) - 1j * epsilon)
    u = field.on_grid[..., 0]
    u_fft = jnp.fft.fftn(u)
    Gu_fft = g_fourier * u_fft
    Gu = jnp.fft.ifftn(Gu_fft)
    return field.replace_params(Gu)
  
  def scattering_potential(
    self,
    field: Field,
    k0=1.0,
    epsilon=0.1
  ) -> Field:
    r"""Implements the scattering potential of the CBS method.

    Args:
      field (FourierSeries): The current field $u$.
      k_sq (FourierSeries): The heterogeneous wavenumber squared.
      k0 (object): The wavenumber.
      epsilon (object): The absorption parameter.

    Returns:
      FourierSeries: The scattering potential.
    """

    k = self.k_sq - k0**2 - 1j * epsilon
    out = field * k
    return out
