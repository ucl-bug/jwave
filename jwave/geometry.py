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

import math
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxdf import Field, FourierSeries
from jaxdf.geometry import Domain
from jaxdf.operators import dot_product, functional
from plum import parametric, type_of

Number = Union[float, int]

@register_pytree_node_class
class Medium:
  r"""
  Medium structure
  Attributes:
    domain (Domain): domain of the medium
    sound_speed (jnp.darray): speed of sound map, can be a scalar
    density (jnp.ndarray): density map, can be a scalar
    attenuation (jnp.ndarray): attenuation map, can be a scalar
    pml_size (int): size of the PML layer in grid-points

  !!! example
    ```python
    N = (128,356)
    medium = Medium(
      sound_speed = jnp.ones(N),
      density = jnp.ones(N),.
      attenuation = 0.0,
      pml_size = 15
    )
    ```
  """
  domain: Domain
  sound_speed: Union[Number, Field] = 1.0
  density: Union[Number, Field] = 1.0
  attenuation: Union[Number, Field] = 0.0
  pml_size: Number = 20.0

  @property
  def int_pml_size(self) -> int:
    return int(self.pml_size)

  def __init__(
    self, domain, sound_speed = 1.0, density=1.0, attenuation=0.0, pml_size=20
  ):
    # Check that all domains are the same
    for field in [sound_speed, density, attenuation]:
      if isinstance(field, Field):
        assert domain == field.domain, "All domains must be the same"

    # Set the attributes
    self.domain = domain
    self.sound_speed = sound_speed
    self.density = density
    self.attenuation = attenuation
    self.pml_size = pml_size

  def tree_flatten(self):
    children = (self.sound_speed, self.density, self.attenuation)
    aux = (self.domain, self.pml_size)
    return (children, aux)

  @classmethod
  def tree_unflatten(cls, aux, children):
    sound_speed, density, attenuation = children
    domain, pml_size = aux
    a = cls(domain, sound_speed, density, attenuation, pml_size)
    return a

  def __str__(self) -> str:
    return self.__repr__()

  def __repr__(self) -> str:
    def show_param(pname):
      attr = getattr(self, pname)
      return f'{pname}: ' + str(attr)
    all_params = sorted(['domain', 'sound_speed','density','attenuation', 'pml_size'])
    strings = list(map(lambda x: show_param(x), all_params))
    return 'Medium:\n - ' + '\n - '.join(strings)



def _points_on_circle(n, radius, centre, cast_int=True, angle=0.0, max_angle=2*np.pi):
  angles = np.linspace(0, max_angle, n, endpoint=False)
  x = (radius * np.cos(angles + angle) + centre[0]).tolist()
  y = (radius * np.sin(angles + angle) + centre[1]).tolist()
  if cast_int:
    x = list(map(int, x))
    y = list(map(int, y))
  return x, y

@parametric(runtime_type_of=True)
class MediumObject(Medium):
  pass

@type_of.dispatch
def _type_of(m: Medium):
  # TODO: Not sure why mypy fails here
  return MediumObject[type(m.sound_speed), type(m.density), type(m.attenuation)] # type: ignore

def _unit_fibonacci_sphere(samples=128):
  # From https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
  points = []
  phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians
  for i in range(samples):
    y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
    radius = math.sqrt(1 - y * y)  # radius at y
    theta = phi * i  # golden angle increment
    x = math.cos(theta) * radius
    z = math.sin(theta) * radius
    points.append((x, y, z))
  return points


def _fibonacci_sphere(n, radius, centre, cast_int=True):
  points = _unit_fibonacci_sphere(n)
  points = np.array(points)
  points = points * radius + centre
  if cast_int:
    points = points.astype(int)
  return points[:, 0], points[:, 1], points[:, 2]


def _circ_mask(N, radius, centre):
  x, y = np.mgrid[0 : N[0], 0 : N[1]]
  dist_from_centre = np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2)
  mask = (dist_from_centre < radius).astype(int)
  return mask

def _sphere_mask(N, radius, centre):
  x, y, z = np.mgrid[0 : N[0], 0 : N[1], 0 : N[2]]
  dist_from_centre = np.sqrt(
    (x - centre[0]) ** 2 + (y - centre[1]) ** 2 + (z - centre[2]) ** 2
  )
  mask = (dist_from_centre < radius).astype(int)
  return mask


@register_pytree_node_class
class Sources:
  r"""Sources structure
  Attributes:
    positions (Tuple[List[int]): source positions
    signals (List[jnp.ndarray]): source signals
  !!! example
    ```python
    x_pos = [10,20,30,40]
    y_pos = [30,30,30,30]
    signal = jnp.sin(jnp.linspace(0,10,100))
    signals = jnp.stack([signal]*4)
    sources = geometry.Source(positions=(x_pos, y_pos), signals=signals)
    ```
  """
  positions: Tuple[np.ndarray]
  signals: Tuple[jnp.ndarray]
  dt: float
  domain: Domain

  def __init__(self, positions, signals, dt, domain):
    self.positions = positions
    self.signals = signals
    self.dt = dt
    self.domain = domain

  def tree_flatten(self):
    children = (self.signals, self.dt)
    aux = (self.domain, self.positions)
    return (children, aux)

  @classmethod
  def tree_unflatten(cls, aux, children):
    signals, dt = children
    domain, positions = aux
    a = cls(positions, signals, dt, domain)
    return a

  def to_binary_mask(self, N):
    r"""
    Convert sources to binary mask
    Args:
      N (Tuple[int]): grid size

    Returns:
      jnp.ndarray: binary mask
    """
    mask = jnp.zeros(N)
    for i in range(len(self.positions[0])):
      mask = mask.at[self.positions[0][i], self.positions[1][i]].set(1)
    return mask > 0

  def on_grid(self, n):

    src = jnp.zeros(self.domain.N)
    if len(self.signals) == 0:
      return src

    idx = n.astype(jnp.int32)
    signals = self.signals[:, idx]
    src = src.at[self.positions].add(signals)
    return jnp.expand_dims(src, -1)

  @staticmethod
  def no_sources(domain):
    return Sources(positions=([], []), signals=([]), dt=1.0, domain=domain)

@register_pytree_node_class
class DistributedTransducer:
  mask: Field
  signal: jnp.ndarray
  dt: float
  domain: Domain

  def __init__(self, mask, signal, dt, domain):
    self.mask = mask
    self.signal = signal
    self.dt = dt
    self.domain = domain

  def tree_flatten(self):
    children = (self.mask, self.signal, self.dt)
    aux = (self.domain,)
    return (children, aux)

  @classmethod
  def tree_unflatten(cls, aux, children):
    mask, signal, dt = children
    domain = aux[0]
    a = cls(mask, signal, dt, domain)
    return a

  def __call__(self, u: Field):
    # returns the transducer output for the wavefield u
    # (Receive mode)
    return dot_product(self.mask, u)

  def set_signal(self, s):
    return DistributedTransducer(self.mask, s, self.dt, self.domain)

  def set_mask(self, m):
    return DistributedTransducer(m, self.signal, self.dt, m.domain)

  def on_grid(self, n):
    # Returns the wavefield produced by the transducer on the grid
    # (Transmit mode)

    if len(self.signal) == 0:
      return 0.

    idx = n.astype(jnp.int32)
    signal = self.signal[idx]
    return signal*self.mask


def get_line_transducer(
  domain,
  position,
  width,
  angle=0
) -> DistributedTransducer:
  r"""
  Construct a line transducer (2D)
  """
  if angle != 0:
    raise NotImplementedError('Angle not implemented yet')

  # Generate mask
  mask = jnp.zeros(domain.N)
  start_col = (domain.N[1]-width)//2
  end_col = (domain.N[1] + width)//2
  mask = mask.at[position, start_col:end_col].set(1.)
  mask = jnp.expand_dims(mask, -1)
  mask = FourierSeries(mask, domain)
  return DistributedTransducer(mask, [], 0., domain)


@dataclass
class TimeHarmonicSource:
  r"""TimeHarmonicSource dataclass

  Attributes:
    domain (Domain): domain
    amplitude (Field): The complex amplitude field of the sources
    omega (float): The angular frequency of the sources

  """
  amplitude: Field
  omega: Union[float, Field]
  domain: Domain

  def on_grid(self, t=0.0):
    r"""Returns the complex field corresponding to the
    sources distribution at time $`t`$.
    """
    return self.amplitude * jnp.exp(1j * self.omega * t)

  @staticmethod
  def from_point_sources(domain, x, y, value, omega):
    src_field = jnp.zeros(domain.N, dtype=jnp.complex64)
    src_field = src_field.at[x, y].set(value)
    return TimeHarmonicSource(src_field, omega, domain)


@register_pytree_node_class
class Sensors:
  r"""Sensors structure
  Attributes:
    positions (Tuple[List[int]]): sensors positions

  !!! example
    ```python
    x_pos = [10,20,30,40]
    y_pos = [30,30,30,30]
    sensors = geometry.Sensors(positions=(x_pos, y_pos))
    ```
  """

  positions: Tuple[tuple]

  def __init__(self, positions):
    self.positions = positions

  def tree_flatten(self):
    children = None
    aux = (self.positions,)
    return (children, aux)

  @classmethod
  def tree_unflatten(cls, aux, children):
    positions = aux[0]
    return cls(positions)

  def to_binary_mask(self, N):
    r"""
    Convert sensors to binary mask
    Args:
      N (Tuple[int]): grid size

    Returns:
      jnp.ndarray: binary mask
    """
    mask = jnp.zeros(N)
    for i in range(len(self.positions[0])):
      mask = mask.at[self.positions[0][i], self.positions[1][i]].set(1)
    return mask > 0

  def __call__(self, p: Field, u: Field, rho: Field ):
    r"""Returns the values of the field u at the sensors positions.
    Args:
      u (Field): The field to be sampled.
    """
    if len(self.positions) == 1:
      return p.on_grid[self.positions[0]]
    elif len(self.positions) == 2:
      return p.on_grid[self.positions[0], self.positions[1]] # type: ignore
    elif len(self.positions) == 3:
      return p.on_grid[self.positions[0], self.positions[1], self.positions[2]] # type: ignore
    else:
      raise ValueError("Sensors positions must be 1, 2 or 3 dimensional. Not {}".format(
        len(self.positions)
      ))

@register_pytree_node_class
class TimeAxis:
  r"""Temporal vector to be used for acoustic
  simulation based on the pseudospectral method of
  [k-Wave](http://www.k-wave.org/)
  Attributes:
    dt (float): time step
    t_end (float): simulation end time
  """
  dt: float
  t_end: float

  def __init__(self, dt, t_end):
    self.dt = dt
    self.t_end = t_end

  def tree_flatten(self):
    children = (None, )
    aux = (self.dt, self.t_end)
    return (children, aux)

  @classmethod
  def tree_unflatten(cls, aux, children):
    dt, t_end = aux
    return cls(dt, t_end)

  @property
  def Nt(self):
    r"""Returns the number of time steps"""
    return np.ceil(self.t_end/self.dt)

  def to_array(self):
    r"""Returns the time-axis as an array"""
    out_steps = jnp.arange(0, self.Nt, 1)
    return out_steps*self.dt

  @staticmethod
  def from_medium(medium: Medium, cfl: float = 0.3, t_end=None):
    r"""Construct a `TimeAxis` object from `kGrid` and `Medium`
    Args:
      grid (kGrid):
      medium (Medium):
      cfl (float, optional):  The [CFL number](http://www.k-wave.org/). Defaults to 0.3.
      t_end ([float], optional):  The final simulation time. If None,
          it is automatically calculated as the time required to travel
          from one corner of the domain to the opposite one.
    """
    dt = cfl * min(medium.domain.dx) / functional(medium.sound_speed)(np.max)
    if t_end is None:
      t_end = np.sqrt(
        sum((x[-1] - x[0]) ** 2 for x in medium.domain.spatial_axis)
      ) / functional(medium.sound_speed)(np.min)
    return TimeAxis(dt=float(dt), t_end=float(t_end))
