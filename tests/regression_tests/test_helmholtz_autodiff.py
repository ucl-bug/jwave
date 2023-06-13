import os
from functools import partial
from typing import Tuple

import numpy as np
import pytest
from jax import device_put, devices, jit, grad, value_and_grad
from jax import numpy as jnp
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat

from jwave import FourierSeries
from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave.geometry import Domain, Medium
from jwave.utils import plot_comparison

RELATIVE_TOLERANCE = 1e-4
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def _make_filename(N, dx, sound_speed, density, attenuation, omega):
  N = str(N).replace(" ", "_")
  return os.path.join(
    DIR_PATH, 
    "..", 
    "regression_data", 
    f"helmholtz_autodiff_{N}_{dx}_{sound_speed}_{density}_{attenuation}_{omega}.mat".replace(" ", "_")
  )


def _index_in_middle(N, span=7):
  return tuple(slice(Ni//2 - span, Ni//2 + span) for Ni in N)


def _get_sos(kind, domain):
  match kind:
    case "scalar":
      return 1500.
    case "homogeneous":
      return FourierSeries(np.ones(domain.N) * 1500., domain)
    case "heterogeneous":
      c = np.ones(domain.N) * 1500.
      c[_index_in_middle(domain.N)] = 2000.
      return FourierSeries(c, domain)
  
  
def _get_density(kind, domain):
  match kind:
    case "scalar":
      return 1000.
    case "homogeneous":
      return FourierSeries(np.ones(domain.N) * 1000., domain)
    case "heterogeneous":
      rho = np.ones(domain.N) * 1000.
      rho[_index_in_middle(domain.N)] = 2000.
      return FourierSeries(rho, domain)
    
    
def _get_attenuation(kind, domain):
  match kind:
    case "scalar":
      return 0.1
    case "homogeneous":
      return FourierSeries(np.zeros(domain.N), domain)
    case "heterogeneous":
      alpha = np.zeros(domain.N)
      alpha[_index_in_middle(domain.N)] = 10.
      return FourierSeries(alpha, domain)


@pytest.mark.parametrize("N", [(48,48), (49,47), (32,32,32), (33,31,32)])
@pytest.mark.parametrize("dx", [1e-3])
@pytest.mark.parametrize("sound_speed", ["heterogeneous"])
@pytest.mark.parametrize("density", ["heterogeneous"])
@pytest.mark.parametrize("attenuation", ["heterogeneous"])
@pytest.mark.parametrize("omega", [1e6])
def test_regression_helmholtz(
  N, dx, sound_speed, density, attenuation, omega, reset_regression_data=False
):
  # Setting up simulation
  dx = tuple([dx] * len(N))
  domain = Domain(N, dx)
  filename = _make_filename(N, dx, sound_speed, density, attenuation, omega)

  # Making source map (dirac at center of domain)
  src = jnp.zeros(N, dtype=jnp.complex64)
  src = src.at[tuple(11 for Ni in N)].set(1.)
  src = FourierSeries(src, domain)
  
  # Making medium
  sound_speed = _get_sos(sound_speed, domain)
  density = _get_density(density, domain)
  attenuation = _get_attenuation(attenuation, domain)
  
  # Move everythin to cpu
  cpu = devices("cpu")[0]
  src = device_put(src, cpu)
  sound_speed = device_put(sound_speed, cpu)
  density = device_put(density, cpu)
  attenuation = device_put(attenuation, cpu)
  
  @jit
  @partial(grad, argnums=[0,1,2,3,4], has_aux=True)
  def loss_fn(
    sound_speed: FourierSeries,
    density: FourierSeries,
    attenuation: FourierSeries,
    omega: float,
    src: FourierSeries,
  ):
    # This tries to maximize the amplitude of the field at a given point
    medium = Medium(src.domain, sound_speed, density, attenuation, pml_size=10)
    solution_field = helmholtz_solver(medium, omega, src, tol=1e-5).on_grid
    max_point = [-11] * len(src.domain.N)
    max_point = tuple(max_point)
    return -jnp.sum(jnp.abs(solution_field[max_point])), solution_field
  
  # Get gradients
  gradients, field = loss_fn(
    sound_speed,
    density,
    attenuation,
    omega,
    src
  )
  
  # Make them numpy arrays
  sos_gradient = np.array(gradients[0].on_grid)
  density_gradient = np.array(gradients[1].on_grid)
  attenuation_gradient = np.array(gradients[2].on_grid)
  omega_gradient = np.array(gradients[3])
  src_gradient = np.array(gradients[4].on_grid)
  
  # Reset regression data if needed
  if reset_regression_data:
    field = np.array(field)
    savemat(filename, {
      "sos_gradient": sos_gradient,
      "density_gradient": density_gradient,
      "attenuation_gradient": attenuation_gradient,
      "omega_gradient": omega_gradient,
      "src_gradient": src_gradient,
      "field": field
    })
  
  # Load regression data
  matfile = loadmat(filename)
  
  # Check each one of them
  err_fun = lambda x,y: jnp.amax(jnp.abs(x-y))/ jnp.amax(jnp.abs(y))
  max_rel_error = max([
    err_fun(sos_gradient, matfile["sos_gradient"]),
    err_fun(density_gradient, matfile["density_gradient"]),
    err_fun(attenuation_gradient, matfile["attenuation_gradient"]),
    err_fun(omega_gradient, matfile["omega_gradient"]),
    err_fun(src_gradient, matfile["src_gradient"]),
  ])
  
  # Make sure the solution is the same within a certain tolerance
  print("  Relative max error = ", 100 * max_rel_error, "%")
  
  assert max_rel_error <RELATIVE_TOLERANCE, (
      "Test failed, error above maximum limit of "
      + str(100 *RELATIVE_TOLERANCE)
      + "%"
  )
  