from diffrax import RecursiveCheckpointAdjoint
from jax import jit
from jax import numpy as jnp
from matplotlib import pyplot as plt

from jwave import FourierSeries
from jwave.acoustics import acoustic_solver
from jwave.geometry import Domain, Medium, circ_mask
from jwave.ode import SemiImplicitEulerCorrected, TimeAxis
from jwave.utils import show_field

N, dx = (128, 128), (0.1e-3, 0.1e-3)
domain = Domain(N, dx)

medium = Medium(domain=domain, sound_speed=1500.0)

time_axis = TimeAxis.from_cfl_number(medium, cfl=0.3)
max_steps = int(time_axis.Nt + 1)

p0 = 1.0 * jnp.expand_dims(circ_mask(N, 4, (80, 60)), -1)
p0 = FourierSeries(p0, domain)


@jit
def compiled_simulator(medium, p0):
    return acoustic_solver(medium,
                           time_axis,
                           SemiImplicitEulerCorrected(),
                           p0=p0,
                           adjoint=RecursiveCheckpointAdjoint())


pressure = compiled_simulator(medium, p0)

t = 250
show_field(pressure[t])
plt.title(f"Pressure field at t={time_axis.to_array()[t]}")
plt.show()
