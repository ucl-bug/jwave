
import os
from functools import partial

import numpy as np
import pytest
from jax import device_put, devices, jit
from jax import numpy as jnp
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat

from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis, _circ_mask
from jwave.utils import plot_comparison


# Setting source
def _get_p0(domain):
  Nx = domain.N
  p0 = 5.0 * _circ_mask(Nx, 5, (40, 40))
  p0 =  jnp.expand_dims(p0, -1)
  p0 = FourierSeries(p0, domain)
  return p0

# Setting sound speed
def _get_heterog_sound_speed(domain):
  sound_speed = np.ones(domain.N) * 1500.0
  sound_speed[50:90, 65:100] = 2300.0
  sound_speed = FourierSeries(np.expand_dims(sound_speed, -1), domain)
  return sound_speed

def _get_homog_sound_speed(domain):
  return 1500.0

# Setting density
def _get_heterog_density(domain):
  density = np.ones(domain.N) * 1000.0
  density[20:40, 65:100] = 2000.0
  density = FourierSeries(np.expand_dims(density, -1), domain)
  return density

def _get_homog_density(domain):
  return 1000.0


TEST_SETTINGS = {
  "ivp_no_pml_no_smooth_homog": {
    "N": (128, 128),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": False,
    "PMLSize": 0,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_homog_sound_speed,
    "rho0_constructor": _get_homog_density,
    "max_err": 1e-5,
  },
  "ivp_pml_no_smooth_homog" : {
    "N": (128, 128),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": False,
    "PMLSize": 10,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_homog_sound_speed,
    "rho0_constructor": _get_homog_density,
    "max_err": 1e-5,
  },
  "ivp_no_pml_smooth_homog": {
    "N": (128, 128),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": True,
    "PMLSize": 0,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_homog_sound_speed,
    "rho0_constructor": _get_homog_density,
    "max_err": 1e-2,
  },
  "ivp_pml_smooth_homog": {
    "N": (128, 128),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": True,
    "PMLSize": 10,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_homog_sound_speed,
    "rho0_constructor": _get_homog_density,
    "max_err": 1e-2,
  },
  "ivp_no_pml_no_smooth_heterog_c0": {
    "N": (128, 128),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": False,
    "PMLSize": 0,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_heterog_sound_speed,
    "rho0_constructor": _get_homog_density,
    "max_err": 1e-5,
  },
  "ivp_no_pml_no_smooth_heterog_rho0": {
    "N": (128, 128),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": False,
    "PMLSize": 0,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_homog_sound_speed,
    "rho0_constructor": _get_heterog_density,
    "max_err": 1e-5,
  },
  "ivp_no_pml_no_smooth_heterog_c0_rho0": {
    "N": (128, 128),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": False,
    "PMLSize": 0,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_heterog_sound_speed,
    "rho0_constructor": _get_heterog_density,
    "max_err": 1e-5,
  },
  "ivp_no_pml_no_smooth_homog_odd": {
    "N": (125, 125),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": False,
    "PMLSize": 0,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_homog_sound_speed,
    "rho0_constructor": _get_homog_density,
    "max_err": 1e-5,
  },
  "ivp_pml_no_smooth_homog_odd" : {
    "N": (125, 125),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": False,
    "PMLSize": 10,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_homog_sound_speed,
    "rho0_constructor": _get_homog_density,
    "max_err": 1e-5,
  },
  "ivp_no_pml_smooth_homog_odd": {
    "N": (125, 125),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": True,
    "PMLSize": 0,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_homog_sound_speed,
    "rho0_constructor": _get_homog_density,
    "max_err": 1e-2,
  },
  "ivp_pml_smooth_homog_odd": {
    "N": (125, 125),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": True,
    "PMLSize": 10,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_homog_sound_speed,
    "rho0_constructor": _get_homog_density,
    "max_err": 1e-2,
  },
  "ivp_no_pml_no_smooth_heterog_c0_odd": {
    "N": (125, 125),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": False,
    "PMLSize": 0,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_heterog_sound_speed,
    "rho0_constructor": _get_homog_density,
    "max_err": 1e-5,
  },
  "ivp_no_pml_no_smooth_heterog_rho0_odd": {
    "N": (125, 125),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": False,
    "PMLSize": 0,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_homog_sound_speed,
    "rho0_constructor": _get_heterog_density,
    "max_err": 1e-5,
  },
  "ivp_no_pml_no_smooth_heterog_c0_rho0_odd": {
    "N": (125, 125),
    "dx": (0.1e-3, 0.1e-3),
    "smooth_initial": False,
    "PMLSize": 0,
    "p0_constructor": _get_p0,
    "c0_constructor": _get_heterog_sound_speed,
    "rho0_constructor": _get_heterog_density,
    "max_err": 1e-5,
  }
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
  time_axis = TimeAxis.from_medium(medium, cfl=0.5, t_end=5e-6)

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

    if isinstance(sound_speed, FourierSeries):
      sound_speed = sound_speed.on_grid

    if isinstance(density, FourierSeries):
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
  err = abs(p_final - kwave_p_final)

  # Check maximum error
  maxErr = jnp.amax(err)
  print('Test name: ' + test_name)
  print('  Maximum error = ', maxErr)
  assert maxErr < settings["max_err"] #, "Test failed, error above maximum limit of " + str(settings["max_err"])
  print('  Test pass')

  if use_plots:
    plot_comparison(p_final, kwave_p_final, test_name, ['j-Wave', 'k-Wave'])
    plt.show()

if __name__ == "__main__":
  for key in TEST_SETTINGS:
    test_ivp(key, use_plots = True)
