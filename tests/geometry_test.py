from jwave import geometry
from jwave.utils import assert_pytree_isclose
from jax import numpy as jnp


def test_make_grid():
    N = [8, 7]
    dx = [0.2, 0.1]
    _ = geometry.kGrid.make_grid(N, dx)


def test_make_time_against_kwave():
    # inputs and outputs
    N = [8, 7]
    dx = [0.2, 0.1]
    cfl = 0.2
    t_end = 0.1

    sound_speed = jnp.ones(N) * 15

    medium = geometry.Medium(
        sound_speed=sound_speed, density=1.0, attenuation=0, pml_size=0
    )

    reference_time = geometry.TimeAxis(dt=0.00133333333333333, t_end=0.1)

    # Check
    grid = geometry.kGrid.make_grid(N, dx)
    t = geometry.TimeAxis.from_kgrid(grid, medium, cfl, t_end)

    assert_pytree_isclose(t, reference_time)


if __name__ == "__main__":
    test_make_grid()
    test_make_time_against_kwave()
