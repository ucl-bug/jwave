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

import warnings

import matplotlib
import numpy as np

from jwave.utils import display_complex_field, is_numeric, plot_complex_field


def test_deprecation_warning():
    """
    Test if deprecation warning is raised when calling plot_complex_field
    """
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Call the deprecated function
        plot_complex_field(
            np.random.rand(10, 10) + 1j * np.random.rand(10, 10))
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "plot_complex_field is deprecated" in str(w[-1].message)


def test_display_complex_field():
    """
    Test if display_complex_field returns correct output types
    """
    fig, axes = display_complex_field(
        np.random.rand(10, 10) + 1j * np.random.rand(10, 10))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(axes, np.ndarray)
    assert all(isinstance(ax, matplotlib.axes.Axes) for ax in axes.flatten())


def test_plot_complex_field():
    """
    Test if plot_complex_field returns correct output types
    """
    fig, axes = plot_complex_field(
        np.random.rand(10, 10) + 1j * np.random.rand(10, 10))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(axes, np.ndarray)
    assert all(isinstance(ax, matplotlib.axes.Axes) for ax in axes.flatten())


def test_is_numeric():
    assert is_numeric(1)
    assert is_numeric(1.0)
    assert is_numeric(1j)
    assert not is_numeric("1")
