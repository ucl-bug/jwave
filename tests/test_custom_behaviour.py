
from jax import jit
from jax import numpy as jnp
from jax import value_and_grad

from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis, _circ_mask

TEST_SETTINGS = {
  "N": (128, 128),
  "dx": (0.1e-3, 0.1e-3),
}


def test_changing_params_wave_prop():
  domain = Domain(
    TEST_SETTINGS["N"], TEST_SETTINGS["dx"]
  )
  medium = Medium(
    domain, 1500., pml_size = 16
  )
  Nx = domain.N
  p0 = 5.0 * _circ_mask(Nx, 5, (40, 40))
  p0 =  jnp.expand_dims(p0, -1)
  p0 = FourierSeries(p0, domain)

  time_axis = TimeAxis.from_medium(medium, cfl=0.5, t_end=5e-6)

  # Get default parameters
  wave_params = simulate_wave_propagation.default_params(
    medium, time_axis
  )

  # Update the pml
  wave_params["pml_u"] = wave_params["pml_u"].replace_params(
    jnp.ones_like(wave_params["pml_u"].on_grid)
  )

  # Run simulation
  @jit
  def run_simulation(params, p0):
    return simulate_wave_propagation(medium, time_axis, p0=p0, params=params)

  _ = run_simulation(wave_params, p0)

def test_differentiating_params():
  domain = Domain(
    TEST_SETTINGS["N"], TEST_SETTINGS["dx"]
  )
  medium = Medium(
    domain, 1500., pml_size = 16
  )
  Nx = domain.N
  p0 = 5.0 * _circ_mask(Nx, 5, (40, 40))
  p0 =  jnp.expand_dims(p0, -1)
  p0 = FourierSeries(p0, domain)

  time_axis = TimeAxis.from_medium(medium, cfl=0.5, t_end=5e-6)
  wave_params = simulate_wave_propagation.default_params(
    medium, time_axis
  )
  wave_params["pml_u"] = wave_params["pml_u"].replace_params(
    jnp.ones_like(wave_params["pml_u"].on_grid)
  )
  true_fields = simulate_wave_propagation(medium, time_axis, p0=p0)

  @jit
  @value_and_grad
  def loss(new_params):
    pred_fields = simulate_wave_propagation(medium, time_axis, p0=p0, params=new_params)
    error = jnp.mean(jnp.abs(pred_fields[-1].on_grid - true_fields[-1].on_grid))
    return error

  error, params_gradients = loss(wave_params)

def test_extract_params_in_jit():

  domain = Domain(
    TEST_SETTINGS["N"], TEST_SETTINGS["dx"]
  )
  medium = Medium(
    domain, 1500., pml_size = 16
  )
  Nx = domain.N
  p0 = 5.0 * _circ_mask(Nx, 5, (40, 40))
  p0 =  jnp.expand_dims(p0, -1)
  p0 = FourierSeries(p0, domain)

  time_axis = TimeAxis.from_medium(medium, cfl=0.5, t_end=5e-6)

  # Run simulation
  @jit
  def run_simulation(to_add, p0):# Get default parameters
    wave_params = simulate_wave_propagation.default_params(
      medium, time_axis
    )
    # Update the pml
    wave_params["pml_u"] = wave_params["pml_u"] + to_add
    return simulate_wave_propagation(medium, time_axis, p0=p0, params=wave_params)

  _ = run_simulation(2., p0)

if __name__ == "__main__":
  test_extract_params_in_jit()
