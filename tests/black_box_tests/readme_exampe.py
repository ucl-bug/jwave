from jax import jit
from jax import numpy as jnp

from jwave import FourierSeries
from jwave.acoustics.time_varying import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis, _circ_mask

# Simulation parameters
domain = Domain(N=(256, 256), dx=(0.1e-3, 0.1e-3))
medium = Medium(domain=domain, sound_speed=1500.)
time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=.8e-05)

# Initial pressure field
p0 = _circ_mask(domain.N, domain.N[0] / 32, (domain.N[0] / 2, domain.N[1] / 2))
p0 = p0 + _circ_mask(domain.N, domain.N[0] / 42, (domain.N[0] / 3, domain.N[1] / 4))
p0 = FourierSeries(jnp.expand_dims(p0,-1), domain)

# Compile and run the simulation
@jit
def solver(medium, p0):
  return simulate_wave_propagation(medium, time_axis, p0=p0)

pressure = solver(medium, p0)

from matplotlib import pyplot as plt

# Save the results
from jwave.utils import show_field

_ = show_field(pressure[250])
plt.title(f'Pressure at time t={time_axis.to_array()[250]}')
plt.savefig('docs/assets/images/readme_example_reconimage.png')
