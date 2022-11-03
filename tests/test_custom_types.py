# This file is part of j-Wave.
#
# j-Wave is free software: you can redistribute it and/or 
# modify it under the terms of the GNU Lesser General Public 
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# j-Wave is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public 
# License along with j-Wave. If not, see <https://www.gnu.org/licenses/>. 

from functools import partial

import pytest
from jax import device_put, devices, jit
from jax import numpy as jnp

from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis


def _uniform_field(value):
  def _initialize_uniform(domain):
    u = jnp.ones(domain.N)*value
    u = jnp.expand_dims(u, -1)
    u = FourierSeries(u, domain)
    return u
  return _initialize_uniform

def _get_value(value):
  def _initialize_value(domain):
    return value
  return _initialize_value

@pytest.mark.parametrize("N", [(64,64), (64, 64, 64)])
@pytest.mark.parametrize("c0", [_get_value(1500.), _uniform_field(1500.)])
@pytest.mark.parametrize("rho0", [_get_value(1000.), _uniform_field(1000.)])
def test_jit_simulate_wave_propagation(
  N,
  c0,
  rho0
):
  dx = [0.1e-3]*len(N)

  # Extract simulation setup
  domain = Domain(N, dx)

  # Empty initial field
  p0 = jnp.zeros(domain.N)
  p0 = jnp.expand_dims(p0, -1)
  p0 = FourierSeries(p0, domain)

  sound_speed = c0(domain)
  density = rho0(domain)

  # Move everything to the CPU
  cpu = devices("cpu")[0]
  sound_speed = device_put(sound_speed, device=cpu)
  density = device_put(density, device=cpu)
  p0 = device_put(p0, device=cpu)

  # Initialize simulation parameters
  medium = Medium(
    domain = domain,
    sound_speed = sound_speed,
    density = density,
    pml_size=10
  )
  time_axis = TimeAxis.from_medium(medium, cfl=0.5, t_end=2e-6)

  # Run simulation
  @partial(jit, backend='cpu')
  def run_simulation(
    p0,
    medium,
    time_axis
    ):
    return simulate_wave_propagation(
      medium,
      time_axis,
      p0=p0
    )

  # Extract last field
  _ = run_simulation(
    p0, medium, time_axis
  )[-1].on_grid[:,:,0]


if __name__ == "__main__":
  test_jit_simulate_wave_propagation(
    N=(64,64),
    c0 = 1500.,
    rho0= 1000.
  )
