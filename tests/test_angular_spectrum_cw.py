import os
from functools import partial

import pytest
from jax import jit
from jax import numpy as jnp
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat

from jwave import FourierSeries
from jwave.acoustics.time_harmonic import angular_spectrum
from jwave.geometry import Domain, Medium
from jwave.utils import plot_comparison

from .utils import log_accuracy


def _test_setter(
  N = (256,256),
  dx = (2e-4,2e-4),
  f0 = 1e6,
  source = "disk",
  c0 = 1480,
  z_pos = 20e-3,
  angular_restriction = True,
  padding = 512,
):
  return {
    "N" : N,
    "dx" : dx,
    "f0" : f0,
    "source" : source,
    "c0" : c0,
    "z_pos" : z_pos,
    "angular_restriction" : angular_restriction,
    "padding" : padding,
  }

def set_source(source_kind, domain):
  if source_kind == "disk":
    r = 20e-3 #domain.dx[0]*domain.N[0]/4
    x,y = domain.spatial_axis
    X, Y = jnp.meshgrid(x,y)
    R = jnp.sqrt(X**2 + Y**2)
    source = jnp.where(R < r, 1.0 + 0j, 0.0j)
    return FourierSeries(source, domain)
  else:
    raise ValueError(f"Unknown source kind: {source_kind}")

TEST_SETTINGS = {
  "angularspectrum_cw_base": _test_setter(),
  "angularspectrum_cw_no_restriction": _test_setter(angular_restriction=False),
  "angularspectrum_cw_close": _test_setter(z_pos=0.1e-3),
  "angularspectrum_cw_far": _test_setter(z_pos=200e-3),
}

@pytest.mark.parametrize("test_name", TEST_SETTINGS.keys())
def test_angular_spectrum_cw(
  test_name,
  use_plots = False,
  reset_mat_file = False
):
  # Initialize field
  settings = TEST_SETTINGS[test_name]
  matfile = test_name + ".mat"
  dir_path = os.path.dirname(os.path.realpath(__file__))
  print(matfile)

  # Setup simulation
  domain = Domain(settings["N"], settings["dx"])
  f0 = settings["f0"]
  source = set_source(settings["source"], domain)
  medium = Medium(
    domain,
    sound_speed=settings["c0"],
  )
  z_pos = settings["z_pos"]
  angular_restriction = settings["angular_restriction"]
  padding = settings["padding"]

  # Evaluate solution with jwave
  @partial(jit, backend="cpu")
  def eval_solution(src_field):
    return angular_spectrum(
      src_field,
      z_pos=z_pos,
      f0=f0,
      medium=medium,
      padding=padding,
      angular_restriction=angular_restriction
    )

  print("Solving with jwave")
  solution_field = eval_solution(source)

  # Generate the matlab results if they don't exist
  if not os.path.isfile(dir_path + '/kwave_data/' + matfile) or reset_mat_file:
    print("Generating matlab results")

    mdict = {
      "pressure": source.on_grid,
      "dx": domain.dx[0],
      "z_pos": z_pos,
      "f0": f0,
      "c0": medium.sound_speed,
      "angular_restriction": angular_restriction,
      "padding": padding,
    }
    in_filepath = dir_path + '/kwave_data/setup_' + matfile
    savemat(in_filepath, mdict)

    mat_command = f"cd('{dir_path}'); test_angular_spectrum_cw(string('{in_filepath}')); exit;"
    command = f'''matlab -nodisplay -nosplash -nodesktop -r "{mat_command}"'''
    os.system(command)

  # Load the matlab results
  out_filepath = dir_path + '/kwave_data/' + matfile
  kwave = loadmat(out_filepath)
  kwave_solution_field = jnp.abs(kwave["p_plane"])
  jwave_solution_field = jnp.abs(solution_field.on_grid[...,0])
  err = abs(jwave_solution_field - kwave_solution_field)

  if use_plots:
    plot_comparison(
      jnp.abs(solution_field.on_grid[...,0]),
      jnp.abs(kwave_solution_field),
      test_name,
      ['j-Wave (abs)', 'k-Wave (abs)'],
      cmap="inferno",
      vmin=0,
    )
    plt.show()

  # Check maximum error
  maxErr = jnp.amax(err)/jnp.amax(kwave_solution_field)
  print('Test name: ' + test_name)
  print('  Maximum error = ', maxErr)
  assert maxErr < 0.01

  # Log error
  log_accuracy(test_name, maxErr)

def test_output_domain_size():
  domain = Domain((64,64), (1,1))
  f0 = 1
  field = FourierSeries(jnp.ones((64,64)) + 0j, domain)
  medium = Medium(
    domain,
    sound_speed=1.,
  )
  out_field = angular_spectrum(
    field,
    z_pos=10,
    f0=f0,
    medium=medium,
    padding=64,
    angular_restriction=True
  )
  assert out_field.domain.N == (64,64)
  assert out_field.domain.N == out_field.on_grid.shape[:-1]

  out_field = angular_spectrum(
    field,
    z_pos=10,
    f0=f0,
    medium=medium,
    padding=64,
    angular_restriction=True,
    unpad_output=False
  )
  assert out_field.domain.N == out_field.on_grid.shape[:-1]
  assert out_field.domain.N == (192,192)
