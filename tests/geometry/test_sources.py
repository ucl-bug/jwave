import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jwave.geometry import Domain, Sources

# Replace 'your_module' with the module where these classes are defined


@pytest.fixture
def domain():
    return Domain(N=(10, 10), dx=(1.0, 1.0))


@pytest.fixture
def sources(domain):
    x_pos = [1, 2, 3, 4]
    y_pos = [3, 3, 3, 3]
    signal = jnp.sin(jnp.linspace(0, 10, 100))
    signals = jnp.stack([signal] * 4)
    sources = Sources(positions=(x_pos, y_pos),
                      signals=signals,
                      dt=0.1,
                      domain=domain)
    return sources


def test_sources_properties(sources):
    assert len(sources.positions[0]) == 4
    assert sources.dt == 0.1
    assert np.array_equal(sources.signals[0], jnp.sin(jnp.linspace(0, 10,
                                                                   100)))


def test_sources_tree_flatten(sources):
    children, aux = sources.tree_flatten()
    assert len(children) == 2
    assert len(aux) == 2


def test_sources_tree_unflatten(domain):
    x_pos = [1, 2, 3, 4]
    y_pos = [3, 3, 3, 3]
    signal = jnp.sin(jnp.linspace(0, 10, 100))
    signals = jnp.stack([signal] * 4)
    sources = Sources.tree_unflatten((domain, (x_pos, y_pos)), (signals, 0.1))
    assert len(sources.positions[0]) == 4
    assert sources.dt == 0.1
    assert np.array_equal(sources.signals[1], jnp.sin(jnp.linspace(0, 10,
                                                                   100)))


def test_sources_to_binary_mask(sources):
    mask = sources.to_binary_mask((10, 10))
    assert mask.shape == (10, 10)
    assert jnp.sum(mask) == 4    # Since we have 4 source positions


def test_sources_on_grid(sources):
    grid = sources.on_grid(jnp.asarray(0))
    assert grid.shape == (10, 10, 1)


def test_sources_no_sources(domain):
    sources = Sources.no_sources(domain)
    assert len(sources.positions[0]) == 0
    assert len(sources.positions[1]) == 0


def test_jit(sources):

    @jax.jit
    def add_dt(sources):
        sources.dt += 1
        return sources.dt

    dt = add_dt(sources)
    assert dt == 1.1
