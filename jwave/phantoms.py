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

from jwave.geometry import circ_mask


def three_circles(N: tuple) -> jnp.ndarray:
    """
    Generate a 3-circle phantom.

    Args:
        N (tuple): The size of the phantom. Must be of length 2.

    Returns:
        jnp.ndarray: The phantom.
    """
    assert len(N) == 2, "N must be of length 2"

    radius = sum(N) / float(len(N))
    mask1 = circ_mask(N, radius * 0.05,
                      (int(N[0] / 2 + N[0] / 8), int(N[1] / 2)))
    mask2 = circ_mask(N, radius * 0.1,
                      (int(N[0] / 2 - N[0] / 8), int(N[1] / 2 + N[1] / 6)))
    mask3 = circ_mask(N, radius * 0.15, (int(N[0] / 2), int(N[1] / 2)))
    p0 = 5.0 * mask1 + 3.0 * mask2 + 4.0 * mask3
    return jnp.expand_dims(p0, -1)
