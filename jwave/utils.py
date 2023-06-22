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

from typing import Tuple, Union

import numpy as np
from jax import numpy as jnp
from jaxdf import Field
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image


def load_image_to_numpy(
    filepath: str,
    padding: int = 0,
    image_size: Tuple[int, int] = None,
) -> np.ndarray:
    r"""Loads an image from a filepath and returns it as a numpy array.

    Args:
        filepath (str): Filepath to the image.
        padding (int, optional): Padding to add to the image. Defaults to 0.
        image_size (Tuple[int, int], optional): Size of the image (excluding padding). Defaults to None.

    Returns:
        np.ndarray: Image as a numpy array.
    """
    img = Image.open(filepath).convert("L")
    img = img.resize(image_size)
    if padding is not None:
        img = np.pad(img, padding, mode="constant")
    return np.array(img).astype(np.float32)


def plot_comparison(
    field1: jnp.ndarray,
    field2: jnp.ndarray,
    title: str = "",
    names: Tuple[str, str] = ("", ""),
    cmap: str = "seismic",
    vmin=None,
    vmax=None,
) -> Figure:
    r"""Plots two 2D fields side by side, and shows the difference between them.

    Args:
        field1 (jnp.ndarray): First field
        field2 (jnp.ndarray): Second Field
        title (str, optional): Title of the plot. Defaults to ''.
        names (Iterable[str], optional): Names of the fields . Defaults to `('','')`.
        cmap (str, optional): Colormap to use. Defaults to 'seismic'.
        vmin (float, optional): Minimum value to use for the colormap. Defaults to None.
        vmax (float, optional): Maximum value to use for the colormap. Defaults to None.

    Returns:
        Figure: Figure object.
    """
    if vmax is None:
        maxval = np.amax(np.abs(field2))
    else:
        maxval = float(vmax)

    if vmin is None:
        minval = -maxval
    else:
        minval = float(vmin)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    plt.suptitle(title)

    im1 = ax1.imshow(field1, vmin=minval, vmax=maxval, cmap=cmap)
    ax1.set_title(names[0])
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)

    im2 = ax2.imshow(field2, vmin=minval, vmax=maxval, cmap=cmap)
    ax2.set_title(names[1])
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2)

    diff = field1 - field2
    maxval = np.amax(np.abs(diff))
    im3 = ax3.imshow(diff, vmin=-maxval, vmax=maxval, cmap="seismic")
    ax3.set_title("Difference")
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3)

    return f


def is_numeric(x):
    """
    Check if x is a numeric value, including complex.
    """
    return isinstance(x, (int, float, complex))


def plot_complex_field(field: Field, figsize=(15, 8), max_intensity=None):
    """
    Plots a complex field.

    Args:
      field (jnp.ndarray): Complex field to plot.
      figsize (tuple): Figure size.
      max_intensity (float): Maximum intensity to plot.
        Defaults to the maximum value in the field.

    Returns:
      matplotlib.pyplot.figure: Figure object.
      matplotlib.pyplot.axes: Axes object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    if isinstance(field, Field):
        field = field.on_grid

    if max_intensity is None:
        max_intensity = jnp.amax(jnp.abs(field))

    axes[0].imshow(field.real,
                   vmin=-max_intensity,
                   vmax=max_intensity,
                   cmap="seismic")
    axes[0].set_title("Real wavefield")
    axes[1].imshow(jnp.abs(field), vmin=0, vmax=max_intensity, cmap="magma")
    axes[1].set_title("Wavefield magnitude")

    return fig, axes


def show_field(
    x: Field,
    title: str = "",
    figsize: Tuple[int, int] = (8, 6),
    vmax: Union[float, int, None] = None,
    aspect: str = "auto",
):
    r"""
    Plots a real valued field. The colormap goes from `-vmax` to `vmax`.

    Args:
      x (Field): Field to plot.
      title (str, optional): Title of the plot. Defaults to "".
      figsize (tuple, optional): Figure size. Defaults to (8,6).
      vmax (float, optional): Maximum value to display. Defaults to None.
      aspect (str, optional): Aspect ratio of the plot. Defaults to "auto".

    Returns:
      matplotlib.pyplot.figure: Figure object.
      matplotlib.pyplot.axes: Axes object.
    """
    if isinstance(x, Field):
        x = x.on_grid

    plt.figure(figsize=figsize)
    maxval = vmax or jnp.amax(jnp.abs(x))
    plt.imshow(
        x,
        cmap="RdBu_r",
        vmin=-maxval,
        vmax=maxval,
        interpolation="nearest",
        aspect=aspect,
    )
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    return None


def show_positive_field(
    x: Field,
    title: str = "",
    figsize: Tuple[int, int] = (8, 6),
    vmax: Union[float, int, None] = None,
    vmin: Union[float, int, None] = None,
    aspect="auto",
):
    if isinstance(x, Field):
        x = x.on_grid
    plt.figure(figsize=figsize)
    if vmax is None:
        vmax = jnp.amax(x)
    if vmin is None:
        vmin = jnp.amin(x)
    plt.imshow(
        x,
        cmap="PuBuGn_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="spline36",
        aspect=aspect,
    )
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    return None
