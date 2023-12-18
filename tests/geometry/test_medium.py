import pytest
from jax import numpy as jnp

from jwave import FourierSeries, OnGrid
from jwave.geometry import Domain, Medium


# Tests for Medium class
def test_medium_type_with_fourier_series():
    domain = Domain((10, ), (1., ))
    fs = FourierSeries(jnp.zeros((10, )), domain)

    m = Medium(domain, sound_speed=fs)
    assert isinstance(
        m, Medium[FourierSeries]), "Type should be Medium[FourierSeries]"

    m = Medium(domain, sound_speed=fs, density=10.0)
    assert isinstance(
        m, Medium[FourierSeries]), "Type should be Medium[FourierSeries]"

    m = Medium(domain)
    assert isinstance(
        m, Medium[FourierSeries]), "Type should be Medium[FourierSeries]"


def test_medium_type_with_on_grid():
    domain = Domain((10, ), (1., ))
    params = jnp.ones((10, ))
    fd = OnGrid(params, domain)

    m = Medium(domain, density=fd)
    assert isinstance(m, Medium[OnGrid]), "Type should be Medium[OnGrid]"


def test_medium_type_mismatch():
    domain = Domain((10, ), (1., ))
    fs = FourierSeries(jnp.zeros((10, )), domain)
    params = jnp.ones((10, ))
    fd = OnGrid(params, domain)

    with pytest.raises(ValueError):
        m = Medium(domain, sound_speed=fs, density=fd)

    with pytest.raises(TypeError):
        m = Medium[int](domain, sound_speed=fs)


if __name__ == "__main__":
    test_medium_type_with_fourier_series()
    test_medium_type_with_on_grid()
    test_medium_type_mismatch()
