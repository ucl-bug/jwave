from jwave import spectral, geometry
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
    da = spectral.derivative(a, grid, 0, 0)(a)
    print(da.shape, reference_da.shape)

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
    da = spectral.derivative(a, grid, 0, 0)(a)
    assert_pytree_isclose(da, reference_da, relative_precision=1e-3, abs_precision=1e-4)


def test_derivative_adjoint():
    rng = jax.random.PRNGKey(2134)
    key1, key2 = jax.random.split(rng)
    N = (8,)
    dx = (0.3,)

    a = jax.random.normal(key1, N)
    b = jax.random.normal(key2, N)

    # Check (d*a,b) = (a,db)
    grid = geometry.kGrid.make_grid(N, dx)

    def d(x):
        return spectral.derivative(x, grid, 0, 0, degree=2)(x)

    _, d_adj_raw = jax.vjp(d, b)

    def d_adj(x):
        return d_adj_raw(x)[0]

    db = d(b)
    d_adj_a = d_adj(a)

    dot_product_1 = jnp.sum(jnp.conj(d_adj_a) * b)
    dot_product_2 = jnp.sum(jnp.conj(a) * db)

    def rel_error(x, y):
        return jnp.abs((x - y) / (x))

    assert rel_error(dot_product_1, dot_product_2) < 1e-4


def test_derivative_with_kspaceop_adjoint():
    rng = jax.random.PRNGKey(2134)
    key1, key2 = jax.random.split(rng)
    N = (8,)
    dx = (0.3,)

    a = jax.random.normal(key1, N)
    b = jax.random.normal(key2, N)

    # Check (d*a,b) = (a,db)
    grid = geometry.kGrid.make_grid(N, dx)
    medium = geometry.Medium(
        sound_speed=jnp.ones(N), density=jnp.ones(N), attenuation=0.0, pml_size=5.0
    )
    time_array = geometry.TimeAxis.from_kgrid(grid, medium, cfl=0.1, t_end=1.0)
    grid = grid.to_staggered()
    grid = grid.apply_kspace_operator(jnp.amin(medium.sound_speed), time_array.dt)

    def d(x):
        return spectral.derivative(x, grid, 0, 0, kspace_op=True)(x)

    _, d_adj_raw = jax.vjp(d, jnp.zeros_like(b))

    def d_adj(x):
        return d_adj_raw(x)[0]

    db = d(b)
    d_adj_a = d_adj(a)

    dot_product_1 = jnp.sum(d_adj_a * b)
    dot_product_2 = jnp.sum(a * db)

    def rel_error(x, y):
        return jnp.abs((x - y) / (x))

    assert rel_error(dot_product_1, dot_product_2) < 1e-4


if __name__ == "__main__":
    test_real_derivative()
    test_complex_derivative()
    test_derivative_adjoint()
    test_derivative_with_kspaceop_adjoint()
