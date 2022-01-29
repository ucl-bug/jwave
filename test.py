import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

from jwave.geometry import Domain, Medium, _circ_mask
from jwave import FourierSeries
from jax import random, jit
from functools import partial
from matplotlib import pyplot as plt
from jax import numpy as jnp
from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave.acoustics.operators import helmholtz
import jax

if __name__ == '__main__':
  import os
  os.environ['CUDA_VISIBLE_DEVICES']='7'
  os.environ['JAX_LOG_COMPILES'] = '1'

  key = random.PRNGKey(42)

  # Defining geometry
  N = (128, 256)         # Grid size
  dx = (1., 1.)          # Spatial resolution
  omega = 1.              # Wavefield omega = 2*pi*f
  target = [160,360]     # Target location

  # Making geometry
  domain = Domain(N, dx)

  # Build the vector that holds the parameters of the apodization an the
  # functions required to transform it into a source wavefield
  transmit_phase = jnp.concatenate([jnp.ones((2,)), jnp.ones((2,))])
  position = list(range(32, 32+4, 2))

  src_field = jnp.zeros(N).astype(jnp.complex64)
  src_field = src_field.at[64, 22].set(1.0)
  src = FourierSeries(jnp.expand_dims(src_field,-1), domain)

  # Constructing medium physical properties
  sound_speed = jnp.zeros(N)
  sound_speed = sound_speed.at[20:105,20:200].set(1.)
  sound_speed = sound_speed*(1-_circ_mask(N, 90,[64,180]))*(1-_circ_mask(N,50,[64,22])) +1
  sound_speed = FourierSeries(jnp.expand_dims(sound_speed,-1), domain)

  medium = Medium(
      domain=domain,
      sound_speed=sound_speed,
      pml_size=15
  )

  @jit
  def solve_helmholtz(medium):
      f = helmholtz_solver(medium, omega, src)
      return f

  field = solve_helmholtz(medium)