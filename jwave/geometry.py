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
from typing import List, Tuple, Union

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxdf import Field, FourierSeries
from jaxdf.geometry import Domain
from jaxdf.mods import Module
from jaxdf.operators import dot_product, functional
from jaxtyping import Array
from plum import parametric

from jwave.logger import logger

Number = Union[float, int]


@parametric
class Medium(Module):
    """_summary_

    Args:
        eqx (_type_): _description_

    Raises:
        ValueError: _description_
        TypeError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    domain: Domain
    sound_speed: Union[Array, Field, float]
    density: Union[Array, Field, float]
    attenuation: Union[Array, Field, float]
    pml_size: float = eqx.field(default=20.0, static=True)

    def __init__(self,
                 domain: Domain,
                 sound_speed: Union[Array, Field, float] = 1.0,
                 density: Union[Array, Field, float] = 1.0,
                 attenuation: Union[Array, Field, float] = 1.0,
                 pml_size: float = 20.0):
        self.domain = domain

        # Check if any input is an Array and none are subclasses of Field
        inputs_are_arrays = [
            isinstance(x, Array) and not jnp.isscalar(x)
            for x in [sound_speed, density, attenuation]
        ]
        inputs_are_fields = [
            issubclass(type(x), Field)
            for x in [sound_speed, density, attenuation]
        ]

        if any(inputs_are_arrays) and any(inputs_are_fields):
            raise ValueError(
                "Ambiguous inputs for Medium: cannot mix Arrays and Field subclasses."
            )

        if all(inputs_are_arrays):
            logger.warning(
                "All inputs are Arrays. This is not recommended for performance reasons. Consider using Fields instead."
            )

        self.sound_speed = sound_speed
        self.density = density
        self.attenuation = attenuation

        # Converting if needed
        for field_name in ["sound_speed", "density", "attenuation"]:
            # Convert to Fourier Series if it is a jax Array and is not a scalar
            if isinstance(
                    self.__dict__[field_name],
                    Array) and not jnp.isscalar(self.__dict__[field_name]):
                #logger.info(f"Converting {field_name}, which is an Array, to a FourierSeries before storing it in the Medium object.")
                self.__dict__[field_name] = FourierSeries(
                    self.__dict__[field_name], domain)

        # Other parameters
        self.pml_size = pml_size

    def __check_init__(self):
        # Check that all domains are the same
        for field in [self.sound_speed, self.density, self.attenuation]:
            if isinstance(field, Field):
                assert self.domain == field.domain, "The domain of all fields must be the same as the domain of the Medium object."

    @classmethod
    def __init_type_parameter__(self, t: type):
        """Check whether the type parameters is valid."""
        if issubclass(t, Field):
            return t
        else:
            raise TypeError(
                f"The type parameter of a Medium object must be a subclass of Field. Got {t}"
            )

    @property
    def max_sound_speed(self):
        """
        Calculate and return the maximum sound speed.

        This property uses the `sound_speed` method/function and applies the `amax`
        function from JAX's numpy (jnp) library to find the maximum sound speed value.

        Returns:
            The maximum sound speed value.
        """
        return functional(self.sound_speed)(jnp.amax)

    @property
    def min_sound_speed(self):
        """
        Calculate and return the minimum sound speed.

        This property uses the `sound_speed` method/function and applies the `amin`
        function from JAX's numpy (jnp) library to find the minimum sound speed value.

        Returns:
            The minimum sound speed value.
        """
        return functional(self.sound_speed)(jnp.amin)

    @property
    def max_density(self):
        """
        Calculate and return the maximum density.

        This property uses the `density` method/function and applies the `amax`
        function from JAX's numpy (jnp) library to find the maximum density value.

        Returns:
            The maximum density value.
        """
        return functional(self.density)(jnp.amax)

    @property
    def min_density(self):
        """
        Calculate and return the minimum density.

        This property uses the `density` method/function and applies the `amin`
        function from JAX's numpy (jnp) library to find the minimum density value.

        Returns:
            The minimum density value.
        """
        return functional(self.density)(jnp.amin)

    @property
    def max_attenuation(self):
        """
        Calculate and return the maximum attenuation.

        This property uses the `attenuation` method/function and applies the `amax`
        function from JAX's numpy (jnp) library to find the maximum attenuation value.

        Returns:
            The maximum attenuation value.
        """
        return functional(self.attenuation)(jnp.amax)

    @property
    def min_attenuation(self):
        """
        Calculate and return the minimum attenuation.

        This property uses the `attenuation` method/function and applies the `amin`
        function from JAX's numpy (jnp) library to find the minimum attenuation value.

        Returns:
            The minimum attenuation value.
        """
        return functional(self.attenuation)(jnp.amin)

    @classmethod
    def __infer_type_parameter__(self, *args, **kwargs):
        """Inter the type parameter from the arguments. Defaults to FourierSeries if
        the parameters are all floats"""
        # Reconstruct kwargs from args
        keys = self.__init__.__code__.co_varnames[1:]
        extra_kwargs = dict(zip(keys, args))
        kwargs.update(extra_kwargs)

        # Get fields types
        field_inputs = ["sound_speed", "density", "attenuation"]
        input_types = []
        for field_name in field_inputs:
            if field_name in kwargs:
                field = kwargs[field_name]

                if isinstance(field, Field):
                    input_types.append(type(field))

        # Keep only unique
        input_types = set(input_types)

        has_fields = len(input_types) > 0
        if not has_fields:
            return FourierSeries

        # Check that there are no more than one field type
        if len(input_types) > 1:
            raise ValueError(
                f"All fields must be of the same type or scalars for a Medium object. Got {input_types}"
            )

        return input_types.pop()

    @classmethod
    def __le_type_parameter__(self, left, right):
        assert len(left) == 1 and len(
            right) == 1, "Medium type parameters can't be tuples."
        return issubclass(left[0], right[0])

    @property
    def int_pml_size(self) -> int:
        r"""Returns the size of the PML layer as an integer"""
        return int(self.pml_size)


def points_on_circle(
        n: int,
        radius: float,
        centre: Tuple[float, float],
        cast_int: bool = True,
        angle: float = 0.0,
        max_angle: float = 2 * np.pi) -> Tuple[List[float], List[float]]:
    """
    Generate points on a circle.

    Args:
        n (int): Number of points.
        radius (float): Radius of the circle.
        centre (tuple): Centre coordinates of the circle (x, y).
        cast_int (bool, optional): If True, points will be rounded and converted to integers. Default is True.
        angle (float, optional): Starting angle in radians. Default is 0.
        max_angle (float, optional): Maximum angle to reach in radians. Default is 2*pi (full circle).

    Returns:
        x, y (tuple): Lists of x and y coordinates of the points.
    """
    angles = np.linspace(0, max_angle, n, endpoint=False)
    x = (radius * np.cos(angles + angle) + centre[0]).tolist()
    y = (radius * np.sin(angles + angle) + centre[1]).tolist()
    if cast_int:
        x = list(map(int, x))
        y = list(map(int, y))
    return x, y


def unit_fibonacci_sphere(
        samples: int = 128) -> List[Tuple[float, float, float]]:
    """
    Generate evenly distributed points on the surface
    of a unit sphere using the Fibonacci Sphere method.

    From https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    Args:
        samples (int, optional): The number of points to generate.
            Default is 128.

    Returns:
        points (list): A list of tuples representing the (x, y, z)
            coordinates of the points on the sphere.
    """
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))    # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2    # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)    # radius at y
        theta = phi * i    # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return points


def fibonacci_sphere(
        n: int,
        radius: float,
        centre: Union[Tuple[float, float, float], np.ndarray],
        cast_int: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate evenly distributed points on the surface of
    a sphere using the Fibonacci Sphere method.

    Args:
    n (int): The number of points to generate.
    radius (float): The radius of the sphere.
    centre (tuple or np.ndarray): The (x, y, z) coordinates of
        the center of the sphere.
    cast_int (bool, optional): If True, points will be rounded
        and converted to integers. Default is True.

    Returns:
    x, y, z (tuple): The x, y, and z coordinates of the points on the sphere.
    """
    points = unit_fibonacci_sphere(n)
    points = np.array(points)
    points = points * radius + centre
    if cast_int:
        points = points.astype(int)
    return points[:, 0], points[:, 1], points[:, 2]


def circ_mask(N: Tuple[int, int], radius: float,
              centre: Union[List[float], Tuple[float, float]]) -> np.ndarray:
    """
    Generate a 2D binary mask representing a circle within a 2D grid.

    The mask is an ndarray of size N with 1s inside the circle (defined by a given
    centre and radius) and 0s outside.

    Args:
        N (Tuple[int, int]): The shape of the output mask (size of the grid).
            It should be in the format (x_size, y_size).
        radius (float): The radius of the circle.
        centre (Union[List[float], Tuple[float, float]]): The coordinates of
            the centre of the circle in the format (x, y).

    Returns:
        mask (np.ndarray): The 2D mask as a numpy ndarray of integers.
            The shape of the mask is N. Values inside the circle are 1, and values
            outside the circle are 0.
    """
    x, y = np.mgrid[0:N[0], 0:N[1]]
    dist_from_centre = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)
    mask = (dist_from_centre < radius).astype(int)
    return mask


def sphere_mask(
        N: Tuple[int, int, int], radius: float,
        centre: Union[List[float], Tuple[float, float, float]]) -> np.ndarray:
    """
    Generate a 3D binary mask representing a sphere within a 3D grid.

    The mask is an ndarray of size N with 1s inside the sphere (defined by a given
    centre and radius) and 0s outside.

    Args:
        N (Tuple[int, int, int]): The shape of the output mask (size of the grid).
            It should be in the format (x_size, y_size, z_size).
        radius (float): The radius of the sphere.
        centre (Union[List[float], Tuple[float, float, float]]): The coordinates of the
            centre of the sphere in the format (x, y, z).

    Returns:
        mask (np.ndarray): The 3D mask as a numpy ndarray of integers. The shape of
            the mask is N. Values inside the sphere are 1, and values outside the
            sphere are 0.
    """
    x, y, z = np.mgrid[0:N[0], 0:N[1], 0:N[2]]
    dist_from_centre = np.sqrt((x - centre[0])**2 + (y - centre[1])**2 +
                               (z - centre[2])**2)
    mask = (dist_from_centre < radius).astype(int)
    return mask


@register_pytree_node_class
class Sources:
    r"""Sources structure

    Attributes:
      positions (Tuple[List[int]): source positions
      signals (List[jnp.ndarray]): source signals
      dt (float): time step
      domain (Domain): domain

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
        aux = (self.domain, )
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
            return 0.0

        idx = n.astype(jnp.int32)
        signal = self.signal[idx]
        return signal * self.mask


def get_line_transducer(domain,
                        position,
                        width,
                        angle=0) -> DistributedTransducer:
    r"""
    Construct a line transducer (2D)
    """
    if angle != 0:
        raise NotImplementedError("Angle not implemented yet")

    # Generate mask
    mask = jnp.zeros(domain.N)
    start_col = (domain.N[1] - width) // 2
    end_col = (domain.N[1] + width) // 2
    mask = mask.at[position, start_col:end_col].set(1.0)
    mask = jnp.expand_dims(mask, -1)
    mask = FourierSeries(mask, domain)
    return DistributedTransducer(mask, [], 0.0, domain)


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
        sources distribution at time $t$.
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
        aux = (self.positions, )
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

    def __call__(self, p: Field, u: Field, rho: Field):
        r"""Returns the values of the field u at the sensors positions.
        Args:
          u (Field): The field to be sampled.
        """
        if len(self.positions) == 1:
            return p.on_grid[self.positions[0]]
        elif len(self.positions) == 2:
            return p.on_grid[self.positions[0],
                             self.positions[1]]    # type: ignore
        elif len(self.positions) == 3:
            return p.on_grid[self.positions[0], self.positions[1],
                             self.positions[2]]    # type: ignore
        else:
            raise ValueError(
                "Sensors positions must be 1, 2 or 3 dimensional. Not {}".
                format(len(self.positions)))


def bli_function(x0: jnp.ndarray,
                 x: jnp.ndarray,
                 n: int,
                 include_imag: bool = False) -> jnp.ndarray:
    """
    The function used to compute the band limited interpolation function.

    Args:
        x0 (jnp.ndarray): Position of the sensors along the axis.
        x (jnp.ndarray): Grid positions.
        n (int): Size of the grid
        include_imag (bool): Include the imaginary component?

    Returns:
        jnp.ndarray: The values of the function at the grid positions.
    """
    dx = jnp.where(
        (x - x0[:, None]) == 0, 1,
        x - x0[:, None])    # https://github.com/google/jax/issues/1052
    dx_nonzero = (x - x0[:, None]) != 0

    if n % 2 == 0:
        y = jnp.sin(jnp.pi * dx) / \
            jnp.tan(jnp.pi * dx / n) / n
        y -= jnp.sin(jnp.pi * x0[:, None]) * jnp.sin(jnp.pi * x) / n
        if include_imag:
            y += 1j * jnp.cos(jnp.pi * x0[:, None]) * jnp.sin(jnp.pi * x) / n
    else:
        y = jnp.sin(jnp.pi * dx) / \
            jnp.sin(jnp.pi * dx / n) / n

    # Deal with case of precisely on grid.
    y = y * jnp.all(dx_nonzero, axis=1)[:, None] + (1 - dx_nonzero) * (
        ~jnp.all(dx_nonzero, axis=1)[:, None])
    return y


@register_pytree_node_class
class BLISensors:
    """ Band-limited interpolant (off-grid) sensors.

    Args:
        positions (Tuple of List of float): Sensor positions.
        n (Tuple of int): Grid size.

    Attributes:
        positions (Tuple[jnp.ndarray]): Sensor positions
        n (Tuple[int]): Grid size.
    """

    positions: Tuple[jnp.ndarray]
    n: Tuple[int]

    def __init__(self, positions: Tuple[jnp.ndarray], n: Tuple[int]):
        self.positions = positions
        self.n = n

        # Calculate the band-limited interpolant weights if not provided.
        x = jnp.arange(n[0])[None]
        self.bx = jnp.expand_dims(bli_function(positions[0], x, n[0]),
                                  axis=range(2, 2 + len(n)))

        if len(n) > 1:
            y = jnp.arange(n[1])[None]
            self.by = jnp.expand_dims(bli_function(positions[1], y, n[1]),
                                      axis=range(2, 2 + len(n) - 1))
        else:
            self.by = None

        if len(n) > 2:
            z = jnp.arange(n[2])[None]
            self.bz = jnp.expand_dims(bli_function(positions[2], z, n[2]),
                                      axis=range(2, 2 + len(n) - 2))
        else:
            self.bz = None

    def tree_flatten(self):
        children = self.positions,
        aux = self.n,
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)

    def __call__(self, p: Field, u, v):
        r"""Returns the values of the field p at the sensors positions.
        Args:
          p (Field): The field to be sampled.
        """
        if len(self.positions) == 1:
            # 1D
            pw = jnp.sum(p.on_grid[None] * self.bx, axis=1)
            return pw
        elif len(self.positions) == 2:
            # 2D
            pw = jnp.sum(p.on_grid[None] * self.bx, axis=1)
            pw = jnp.sum(pw * self.by, axis=1)
            return pw
        elif len(self.positions) == 3:
            # 3D
            pw = jnp.sum(p.on_grid[None] * self.bx, axis=1)
            pw = jnp.sum(pw * self.by, axis=1)
            pw = jnp.sum(pw * self.bz, axis=1)
            return pw
        else:
            raise ValueError(
                "Sensors positions must be 1, 2 or 3 dimensional. Not {}".
                format(len(self.positions)))


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
        return np.ceil(self.t_end / self.dt)

    def to_array(self):
        r"""Returns the time-axis as an array"""
        out_steps = jnp.arange(0, self.Nt, 1)
        return out_steps * self.dt

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
        dt = cfl * min(medium.domain.dx) / functional(medium.sound_speed)(
            np.max)
        if t_end is None:
            t_end = np.sqrt(
                sum((x[-1] - x[0])**2
                    for x in medium.domain.spatial_axis)) / functional(
                        medium.sound_speed)(np.min)
        return TimeAxis(dt=float(dt), t_end=float(t_end))
