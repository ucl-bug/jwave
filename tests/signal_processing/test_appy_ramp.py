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

from jwave.signal_processing import apply_ramp


def test_apply_ramp_with_zero_signal():
    signal = np.zeros(100)
    dt = 0.01
    center_freq = 10.0
    result = apply_ramp(signal, dt, center_freq)
    assert np.all(result == 0)


def test_apply_ramp_with_uniform_signal():
    signal = np.ones(100)
    dt = 0.01
    center_freq = 10.0
    result = apply_ramp(signal, dt, center_freq)
    assert np.allclose(result[:31], np.linspace(0, 1, 31))
    assert np.all(result[31:] == 1)


def test_apply_ramp_with_negative_center_freq():
    signal = np.ones(100)
    dt = 0.01
    center_freq = -10.0
    with pytest.raises(ValueError):
        apply_ramp(signal, dt, center_freq)


def test_apply_ramp_with_zero_center_freq():
    signal = np.ones(100)
    dt = 0.01
    center_freq = 0.0
    with pytest.raises(ValueError):
        apply_ramp(signal, dt, center_freq)


def test_apply_ramp_with_complex_signal():
    signal = np.ones(100) + 1j * np.ones(100)
    dt = 0.01
    center_freq = 10.0
    result = apply_ramp(signal, dt, center_freq)
    expected = (np.ones(100) + 1j * np.ones(100)) * np.concatenate(
        [np.linspace(0, 1, 31), np.ones(69)])
    assert np.allclose(result, expected)


def test_apply_ramp_with_non_1D_signal():
    signal = np.ones((10, 10))
    dt = 0.01
    center_freq = 10.0
    with pytest.raises(ValueError):
        apply_ramp(signal, dt, center_freq)


def test_apply_ramp_with_high_warmup_cycles():
    signal = np.ones(200)
    dt = 0.01
    center_freq = 10.0
    warmup_cycles = 10.0
    result = apply_ramp(signal, dt, center_freq, warmup_cycles)
    assert np.allclose(result[:101], np.linspace(0, 1, 101))
    assert np.all(result[101:] == 1)
