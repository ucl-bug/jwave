from jax import numpy as jnp
from jaxdf import Field
from matplotlib import pyplot as plt


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
  fig, axes = plt.subplots(1 ,2, figsize=figsize)
  field = field.on_grid

  if max_intensity is None:
    max_intensity = jnp.amax(jnp.abs(field))

  axes[0].imshow(field.real, vmin=-max_intensity, vmax=max_intensity, cmap="seismic")
  axes[0].set_title("Real wavefield")
  axes[1].imshow(jnp.abs(field), vmin=0, vmax=max_intensity, cmap="magma")
  axes[1].set_title("Wavefield magnitude")

  return fig, axes


def show_field(x: Field, title="", figsize=(8,6), vmax=None, aspect="auto"):
  if isinstance(x, Field):
    x = x.on_grid

  plt.figure(figsize=figsize)
  maxval = vmax or jnp.amax(jnp.abs(x))
  plt.imshow(
    x,
    cmap="RdBu_r",
    vmin=-maxval,
    vmax=maxval,
    interpolation="spline36",
    aspect=aspect,
  )
  plt.colorbar()
  plt.title(title)
  plt.axis("off")
  return None


def show_positive_field(x: Field, title="", figsize=(8,6), vmax=None, vmin=None, aspect="auto"):
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
