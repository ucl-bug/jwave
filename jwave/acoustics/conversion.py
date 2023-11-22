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

import numpy as np
from jax import numpy as jnp


def db2neper(
    alpha: jnp.ndarray,
    y: jnp.ndarray,
):
    r"""
    Transforms absorption units from decibels to nepers.
    See http://www.k-wave.org/documentation/db2neper.php

    Args:
        alpha(jnp.ndarray): Absorption coefficient in decibels.
        y(jnp.ndarray): Exponent of the absorption coefficient.

    Returns:
        jnp.ndarray: Absorption coefficient in nepers.
    """
    return 100 * alpha * ((1e-6 / (2 * np.pi))**y) / (20 * np.log10(np.exp(1)))
