from jax import numpy as jnp

from jwave.signal_processing import analytic_signal


def test_analytic_signal():
  x = jnp.linspace(0, 8*jnp.pi, 16*8, endpoint=False)
  y = jnp.sin(x)
  y_analytic = analytic_signal(y, -1)
  assert jnp.allclose(jnp.abs(y_analytic), 1.)

if __name__ == "__main__":
  test_analytic_signal()
