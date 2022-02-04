import jax
from jax import jit
from jax import numpy as jnp
from jax import random

from jwave import FourierSeries
from jwave.acoustics.operators import helmholtz
from jwave.geometry import Domain, Medium, _circ_mask

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
def helm_func(medium):
    return helmholtz(src, medium, omega)

field = helm_func(medium)


with jax.profiler.trace('profiling/'):

  # Run the operations to be profiled
  field = helm_func(medium)
  _ = field.params.block_until_ready()

  # Run the operations to be profiled
  field = helm_func(medium)
  _ = field.params.block_until_ready()
