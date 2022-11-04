# This file is part of j-Wave.
#
# j-Wave is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# j-Wave is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with j-Wave. If not, see <https://www.gnu.org/licenses/>.

from jax import numpy as jnp

from jwave.signal_processing import analytic_signal


def test_analytic_signal():
  x = jnp.linspace(0, 8*jnp.pi, 16*8, endpoint=False)
  y = jnp.sin(x)
  y_analytic = analytic_signal(y, -1)
  assert jnp.allclose(jnp.abs(y_analytic), 1.)

if __name__ == "__main__":
  test_analytic_signal()
