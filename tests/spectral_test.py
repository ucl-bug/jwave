from jwave import spectral, geometry
from jwave.geometry import Staggered
from jwave.utils import assert_pytree_isclose
from jax import numpy as jnp
import jax


def test_real_derivative():
    N = [8]
    dx = [0.3]
    a = jnp.array(
        [
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    reference_da = jnp.array([-1.3090, 3.1602, 0, -3.1602, 1.3090, -0.5422, 0, 0.5422])
    grid = geometry.kGrid.make_grid(N, dx)
    D, grid = spectral.derivative_init(a, grid, 0, Staggered.NONE)
    da = D(a, grid, degree=1.)

    assert_pytree_isclose(da, reference_da, relative_precision=1e-3, abs_precision=1e-4)


def test_complex_derivative():

    N = [8]
    dx = [0.3]
    a = jnp.array(
        [
            0.0000 + 0.0000j,
            0.0000 + 0.0000j,
            1.0000 + 0.0000j,
            0.0000 + 1.0000j,
            0.0000 + 0.0000j,
            0.0000 + 0.0000j,
            0.0000 + 0.0000j,
            0.0000 + 0.0000j,
        ]
    )
    reference_da = jnp.array(
        [
            -2.6180 - 0.7668j,
            4.4692 + 0.0000j,
            -1.3090 + 1.8512j,
            -1.8512 + 1.3090j,
            -0.0000 - 4.4692j,
            0.7668 + 2.6180j,
            -1.3090 - 1.8512j,
            1.8512 + 1.3090j,
        ]
    )

    # Check
    grid = geometry.kGrid.make_grid(N, dx)
    D, grid = spectral.derivative_init(a, grid, 0, Staggered.NONE)
    da = D(a, grid, degree=1.)
    assert_pytree_isclose(da, reference_da, relative_precision=1e-3, abs_precision=1e-4)

if __name__ == "__main__":
    test_real_derivative()
    test_complex_derivative()