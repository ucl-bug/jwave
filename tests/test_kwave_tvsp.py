
import os

import pytest
from jax import jit
from jax import numpy as jnp
from scipy.io import loadmat, savemat

from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.signal_processing import gaussian_window

import numpy as np

# Setting source
def _get_sources(domain, time_axis):
  
  t = time_axis.to_array()
  s = jnp.sin(2 * jnp.pi * 4e6 * t)
  
  s1 = gaussian_window(s, t, 0.5e-6, 4e-7)
  s2 = gaussian_window(s, t, 1.5e-6, 4e-7)
  s3 = gaussian_window(s, t, 1.5e-6, 4e-7)
  
  sources = Sources(
    positions=((28, 64, 80), (32, 32, 100)),
    signals=jnp.stack([s1, s2, s3]),
    dt=time_axis.dt,
    domain=domain,
  )
  return sources

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
  "tvsp_no_pml_homog": {
    "N": (128, 128),
    "dx": (0.1e-3, 0.1e-3),
    "PMLSize": 0,
    "source_constructor": _get_sources,
    "c0_constructor": _get_homog_sound_speed,
    "rho0_constructor": _get_homog_density,
    "max_err": 1e-5,
  },
  "tvsp_pml_homog" : {
    "N": (128, 128),
    "dx": (0.1e-3, 0.1e-3),
    "PMLSize": 10,
    "source_constructor": _get_sources,
    "c0_constructor": _get_homog_sound_speed,
    "rho0_constructor": _get_homog_density,
    "max_err": 1e-5,
  },
  "tvsp_no_pml_heterog_c0": {
    "N": (128, 128),
    "dx": (0.1e-3, 0.1e-3),
    "PMLSize": 0,
    "source_constructor": _get_sources,
    "c0_constructor": _get_heterog_sound_speed,
    "rho0_constructor": _get_homog_density,
    "max_err": 1e-5,
  },
  "tvsp_no_pml_heterog_rho0": {
    "N": (128, 128),
    "dx": (0.1e-3, 0.1e-3),
    "PMLSize": 0,
    "source_constructor": _get_sources,
    "c0_constructor": _get_homog_sound_speed,
    "rho0_constructor": _get_heterog_density,
    "max_err": 1e-5,
  },
  "tvsp_no_pml_heterog_c0_rho0": {
    "N": (128, 128),
    "dx": (0.1e-3, 0.1e-3),
    "PMLSize": 0,
    "source_constructor": _get_sources,
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

  # Initialize simulation parameters
  medium = Medium(
    domain = domain,
    sound_speed = sound_speed,
    density = density,
    pml_size=settings["PMLSize"]
  )
  time_axis = TimeAxis.from_medium(medium, cfl=0.5, t_end=4e-6)
  sources = settings["source_constructor"](domain, time_axis)

  # Run simulation
  @jit
  def run_simulation(sources):
    return simulate_wave_propagation(
      medium,
      time_axis,
      sources=sources,
    )

  # Extract last field
  p_final = run_simulation(sources)[-1].on_grid[:,:,0]

  # Generate the matlab results if they don't exist
  if not os.path.isfile(dir_path + '/kwave_data/' + matfile):
    print("Generating matlab results")

    if isinstance(sound_speed, FourierSeries):
      sound_speed = sound_speed.on_grid

    if isinstance(density, FourierSeries):
      density = density.on_grid

    mdict = {
      "p_final": p_final,
      "source_positions": sources.positions,
      "source_signals": sources.signals,
      "Nx": domain.N,
      "dx": domain.dx,
      "Nt": time_axis.Nt,
      "dt": time_axis.dt,
      "sound_speed": sound_speed,
      "density": density,
      "PMLSize": settings["PMLSize"]
    }
    in_filepath = dir_path + '/kwave_data/setup_' + matfile
    savemat(in_filepath, mdict)

    mat_command = f"cd('{dir_path}'); test_kwave_tvsp(string('{in_filepath}')); exit;"
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
  assert maxErr < settings["max_err"], "Test failed, error above maximum limit of " + str(settings["max_err"])
  print('  Test pass')

  if use_plots:
    plot_comparison(p_final, kwave_p_final)


def plot_comparison(jwave, kwave):

  import matplotlib.pyplot as plt
  from mpl_toolkits.axes_grid1 import make_axes_locatable

  plt.rcParams.update({'font.size': 6})
  plt.rcParams["figure.dpi"] = 300

  f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

  im1 = ax1.imshow(jwave)
  ax1.set_title('j-Wave')
  divider1 = make_axes_locatable(ax1)
  cax1 = divider1.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(im1, cax=cax1)

  im2 = ax2.imshow(kwave)
  ax2.set_title('k-Wave')
  divider2 = make_axes_locatable(ax2)
  cax2 = divider2.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(im2, cax=cax2)

  im3 = ax3.imshow((jwave - kwave))
  ax3.set_title('Difference')
  divider3 = make_axes_locatable(ax3)
  cax3 = divider3.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(im3, cax=cax3)

  plt.show()


if __name__ == "__main__":
  for key in TEST_SETTINGS:
    test_ivp(key, use_plots = True)
