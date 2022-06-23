
import os
from functools import partial
from typing import Tuple

import numpy as np
import pytest
from jax import device_put, devices, jit
from jax import numpy as jnp
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat

from jwave import FiniteDifferences
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis, _circ_mask
from jwave.utils import plot_comparison

from .utils import log_accuracy

# Default figure settings
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.dpi"] = 300


# Setting source
def _get_p0(domain):
  Nx = domain.N
  p0 = 5.0 * _circ_mask(Nx, 5, (40, 40))
  p0 =  jnp.expand_dims(p0, -1)
  p0 = FiniteDifferences(p0, domain, accuracy=8)
  return p0

# Setting sound speed
def _get_heterog_sound_speed(domain):
  sound_speed = np.ones(domain.N) * 1500.0
  sound_speed[50:90, 32:100] = 2300.0
  sound_speed = FiniteDifferences(np.expand_dims(sound_speed, -1), domain, accuracy=8)
  return sound_speed

def _get_homog_sound_speed(domain):
  sound_speed = np.ones(domain.N) * 1500.0
  sound_speed = FiniteDifferences(np.expand_dims(sound_speed, -1), domain, accuracy=8)
  return sound_speed

# Setting density
def _get_heterog_density(domain):
  density = np.ones(domain.N) * 1000.0
  density[20:40, 65:100] = 2000.0
  density = FiniteDifferences(np.expand_dims(density, -1), domain, accuracy=8)
  return density

def _get_homog_density(domain):
  return 1000.0

def _test_setter(
  N: Tuple[int] = (128,128),
  dx = 0.1e-3,
  smooth_initial: bool = True,
  PMLSize: int = 0,
  p0_constructor = _get_p0,
  c0_constructor = _get_homog_sound_speed,
  rho0_constructor = _get_homog_density,
  max_err = 0.05,
):
  dx = tuple([dx]*len(N))
  return {
    "N" : N,
    "dx" : dx,
    "smooth_initial" : smooth_initial,
    "PMLSize" : PMLSize,
    "p0_constructor" : p0_constructor,
    "c0_constructor" : c0_constructor,
    "rho0_constructor" : rho0_constructor,
    "max_err" : max_err,
  }

TEST_SETTINGS = {
  "ivp_fd_no_pml": _test_setter(),
  "ivp_fd_pml": _test_setter(
    PMLSize = 16,
    max_err = 0.1,
  ),
  "ivp_fd_heterog_c0": _test_setter(
    PMLSize = 16,
    c0_constructor = _get_heterog_sound_speed,
    max_err=0.2
  ),
  "ivp_fd_heterog_rho0": _test_setter(
    PMLSize = 16,
    rho0_constructor = _get_heterog_density,
    max_err=0.2
  ),
  "ivp_fd_wide_heterog_rho0": _test_setter(
    N = (128,192),
    PMLSize = 16,
    rho0_constructor = _get_heterog_density,
    max_err=0.2
  )
}


@pytest.mark.parametrize("test_name", TEST_SETTINGS.keys())
def test_ivp(
  test_name,
  use_plots = False
):
  settings = TEST_SETTINGS[test_name]
  matfile = test_name + ".mat"
  dir_path = os.path.dirname(os.path.realpath(__file__))

  # Extract simulation setup
  domain = Domain(settings["N"], settings["dx"])
  sound_speed = settings["c0_constructor"](domain)
  density = settings["rho0_constructor"](domain)
  p0 = settings["p0_constructor"](domain)

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
    pml_size=settings["PMLSize"]
  )
  time_axis = TimeAxis.from_medium(medium, cfl=0.1, t_end=4e-6)

  # Run simulation
  @partial(jit, backend='cpu')
  def run_simulation(p0):
    return simulate_wave_propagation(
      medium,
      time_axis,
      p0=p0,
      smooth_initial=settings["smooth_initial"]
    )

  # Extract last field
  p_final = run_simulation(p0)[-1].on_grid[:,:,0]

  # Generate the matlab results if they don't exist
  if not os.path.isfile(dir_path + '/kwave_data/' + matfile):
    print("Generating matlab results")

    if isinstance(sound_speed, FiniteDifferences):
      sound_speed = sound_speed.on_grid

    if isinstance(density, FiniteDifferences):
      density = density.on_grid

    mdict = {
      "p_final": p_final,
      "p0": p0.on_grid[...,0],
      "Nx": domain.N,
      "dx": domain.dx,
      "Nt": time_axis.Nt,
      "dt": time_axis.dt,
      "sound_speed": sound_speed,
      "density": density,
      "PMLSize": settings["PMLSize"],
      "smooth_initial": settings["smooth_initial"]
    }
    in_filepath = dir_path + '/kwave_data/setup_' + matfile
    savemat(in_filepath, mdict)

    fun_call = f'''test_kwave_ivp('{in_filepath}')'''
    mat_command = f"cd('{dir_path}'); test_kwave_ivp(string('{in_filepath}')); exit;"
    command = f'''matlab -nodisplay -nosplash -nodesktop -r "{mat_command}"'''
    os.system(command)

  # Load the matlab results
  out_filepath = dir_path + '/kwave_data/' + matfile
  kwave = loadmat(out_filepath)
  kwave_p_final = kwave["p_final"]
  err = abs(p_final - kwave_p_final) / jnp.amax(abs(p_final))

  if use_plots:
    plot_comparison(p_final, kwave_p_final, test_name, ['j-Wave', 'k-Wave'])
    plt.show()

  # Check maximum error
  maxErr = jnp.amax(err)/jnp.amax(jnp.abs(kwave_p_final))
  print('Test name: ' + test_name)
  print('  Maximum error = ', 100*maxErr, "%")
  assert maxErr < settings["max_err"] #, "Test failed, error above maximum limit of " + str(settings["max_err"])

  # Log error
  log_accuracy(test_name, maxErr)

if __name__ == "__main__":
  for key in TEST_SETTINGS:
    test_ivp(key, use_plots = True)
