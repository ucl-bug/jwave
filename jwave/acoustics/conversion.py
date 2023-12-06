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
from typing import Union
from jaxtyping import ArrayLike

def db2neper(
    alpha: ArrayLike,
    y: ArrayLike = 1,
):
    r"""
    Transforms absorption units from decibels to nepers.
    See http://www.k-wave.org/documentation/db2neper.php

    Args:
        alpha(ArrayLike): Absorption coefficient in decibels.
        y(ArrayLike): Exponent of the absorption coefficient.

    Returns:
        ArrayLike: Absorption coefficient in nepers.
    """
    return 100 * alpha * ((1e-6 / (2 * np.pi))**y) / (20 * np.log10(np.exp(1)))

def neper2db(
    alpha: ArrayLike,
    y: ArrayLike = 1
):  
    r"""
    Transforms absorption units from nepers to decibels.
    See http://www.k-wave.org/documentation/neper2db.php
    
    Args:
        alpha(ArrayLike): Absorption coefficient in nepers.
        y(ArrayLike): Exponent of the absorption coefficient.
        
    Returns:
        ArrayLike: Absorption coefficient in decibels.
    """
    return 20 * np.log10(np.exp(1)) * alpha * (2 * np.pi * 1e6)**y / 100