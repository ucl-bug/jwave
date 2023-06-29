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
import pytest
from jax import numpy as jnp

from jwave.signal_processing import fourier_downsample, fourier_upsample


@pytest.mark.parametrize(
    "input_array, subsample, discard_last, expected_shape",
    [
    # Test with 2D array, even shape
        (np.random.rand(4, 4), 2, False, (2, 2)),
    # Test with 2D array, odd shape
        (np.random.rand(5, 5), 2, False, (3, 3)),
    # Test with 3D array, even shape
        (np.random.rand(4, 4, 4), 2, True, (2, 2, 4)),
    # Test with 3D array, odd shape
        (np.random.rand(5, 5, 5), 2, False, (3, 3, 3)),
    ])
def test_fourier_downsample(input_array, subsample, discard_last,
                            expected_shape):
    jax_input_array = jnp.array(input_array)
    result = fourier_downsample(jax_input_array, subsample, discard_last)
    assert result.shape == expected_shape


def test_fourier_downsample_subsample_one():
    input_array = np.random.rand(4, 4)
    jax_input_array = jnp.array(input_array)
    result = fourier_downsample(jax_input_array, subsample=1)
    assert jnp.array_equal(result, jax_input_array)


@pytest.mark.parametrize(
    "input_array, upsample, discard_last, expected_shape",
    [
    # Test with 2D array, even shape
        (np.random.rand(2, 2), 2, False, (4, 4)),
    # Test with 2D array, odd shape
        (np.random.rand(3, 3), 2, False, (6, 6)),
    # Test with 3D array, even shape
        (np.random.rand(2, 2, 2), 2, True, (4, 4, 2)),
    # Test with 3D array, even shape
        (np.random.rand(2, 2, 2), 2, False, (4, 4, 4)),
    # Test with 3D array, odd shape
        (np.random.rand(3, 3, 3), 2, False, (6, 6, 6)),
    ])
def test_fourier_upsample(input_array, upsample, discard_last, expected_shape):
    jax_input_array = jnp.array(input_array)
    result = fourier_upsample(jax_input_array, upsample, discard_last)
    assert result.shape == expected_shape


def test_fourier_upsample_upsample_one():
    input_array = np.random.rand(4, 4)
    jax_input_array = jnp.array(input_array)
    result = fourier_upsample(jax_input_array, upsample=1)
    assert jnp.array_equal(result, jax_input_array)
