import os

import numpy as np
import pytest
from jax import device_put, devices, jit
from jax import numpy as jnp
from scipy.io import loadmat, savemat

from jwave import FourierSeries
from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave.geometry import Domain, Medium

RELATIVE_TOLERANCE = 1e-4
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def _make_filename(N, dx, sound_speed, density, attenuation, omega):
    N = str(N).replace(" ", "_")
    return os.path.join(
        DIR_PATH,
        "..",
        "regression_data",
        f"helmholtz_{N}_{dx}_{sound_speed}_{density}_{attenuation}_{omega}.mat"
        .replace(" ", "_"),
    )


def _index_in_middle(N, span=7):
    return tuple(slice(Ni // 2 - span, Ni // 2 + span) for Ni in N)


def _get_sos(kind, domain):
    match kind:
        case "scalar":
            return 1500.0
        case "homogeneous":
            return FourierSeries(np.ones(domain.N) * 1500.0, domain)
        case "heterogeneous":
            c = np.ones(domain.N) * 1500.0
            c[_index_in_middle(domain.N)] = 2000.0
            return FourierSeries(c, domain)


def _get_density(kind, domain):
    match kind:
        case "scalar":
            return 1000.0
        case "homogeneous":
            return FourierSeries(np.ones(domain.N) * 1000.0, domain)
        case "heterogeneous":
            rho = np.ones(domain.N) * 1000.0
            rho[_index_in_middle(domain.N)] = 2000.0
            return FourierSeries(rho, domain)


def _get_attenuation(kind, domain):
    match kind:
        case "scalar":
            return 0.1
        case "homogeneous":
            return FourierSeries(np.zeros(domain.N), domain)
        case "heterogeneous":
            alpha = np.zeros(domain.N)
            alpha[_index_in_middle(domain.N)] = 10.0
            return FourierSeries(alpha, domain)


@pytest.mark.parametrize("N", [(32, 32), (33, 31), (32, 32, 32), (33, 31, 32)])
@pytest.mark.parametrize("dx", [1e-3])
@pytest.mark.parametrize("sound_speed", ["scalar", "heterogeneous"])
@pytest.mark.parametrize("density", ["scalar", "heterogeneous"])
@pytest.mark.parametrize("attenuation", ["scalar", "heterogeneous"])
@pytest.mark.parametrize("omega", [1.5e6])
def test_regression_helmholtz(N,
                              dx,
                              sound_speed,
                              density,
                              attenuation,
                              omega,
                              reset_regression_data=False):
    # Setting up simulation
    dx = tuple([dx] * len(N))
    domain = Domain(N, dx)
    filename = _make_filename(N, dx, sound_speed, density, attenuation, omega)

    # Making source map (dirac at center of domain)
    src = jnp.zeros(N, dtype=jnp.complex64)
    src = src.at[tuple(Ni // 2 for Ni in N)].set(1.0)
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

    # Initialize medium
    medium = Medium(domain, sound_speed, density, attenuation, pml_size=10)

    # Run the simulation
    @jit
    def run_simulation(medium, src):
        return helmholtz_solver(medium, omega, src, tol=1e-5)

    # Get field
    solution_field = run_simulation(medium, src)

    # Reset regression data if needed
    if reset_regression_data:
        sol_as_numpy = np.array(solution_field.on_grid)
        savemat(filename, {"solution_field": sol_as_numpy})

    # Load regression data
    previous_solution = loadmat(filename)["solution_field"]

    # Make sure the solution is the same within a certain tolerance
    err = jnp.abs(solution_field.on_grid - previous_solution)
    relErr = jnp.amax(err) / jnp.amax(jnp.abs(previous_solution))
    print("  Relative max error = ", 100 * relErr, "%")

    assert relErr < RELATIVE_TOLERANCE, (
        "Test failed, error above maximum limit of " +
        str(100 * RELATIVE_TOLERANCE) + "%")
