import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from jax import numpy as jnp
from jaxdf import Field
from jaxdf.geometry import Domain


@dataclass(init=False)
class Medium:
    r"""
    Medium structure
    Attributes:
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
    sound_speed: jnp.ndarray
    density: jnp.ndarray
    attenuation: jnp.ndarray
    pml_size: float = 20

    def __init__(
        self, domain, sound_speed, density=None, attenuation=None, pml_size=20
    ):
        self.domain = domain
        self.sound_speed = sound_speed

        if density is None:
            self.density = jnp.ones_like(sound_speed)
        else:
            self.density = density

        # if attenuation is None:
        #    self.attenuation = 0
        # else:
        #    self.attenuation = attenuation
        self.attenuation = attenuation

        self.pml_size = pml_size


def _points_on_circle(n, radius, centre, cast_int=True, angle=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = (radius * np.cos(angles + angle) + centre[0]).tolist()
    y = (radius * np.sin(angles + angle) + centre[1]).tolist()
    if cast_int:
        x = list(map(int, x))
        y = list(map(int, y))
    return x, y


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


@dataclass
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
    positions: Tuple[jnp.ndarray]
    signals: Tuple[jnp.ndarray]
    dt: float
    domain: Domain

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

    def to_field(self, t):

        src = jnp.zeros(self.domain.N)
        if len(self.signals) == 0:
            return src

        idx = (t / self.dt).round().astype(jnp.int32)
        signals = self.source_signals[:, idx] / len(self.domain.N)
        src = src.at[self.positions].add(signals)
        return jnp.expand_dims(src, -1)

    @staticmethod
    def no_sources(domain):
        return Sources(positions=([], []), signals=([]), dt=1.0, domain=domain)


@dataclass
class TimeHarmonicSource:
    r"""TimeHarmonicSource structure

        Attributes:
          amplitude (float): amplitude of the source


        !!! example
            ```python
            x_pos = [10,20,30,40]
            y_pos = [30,30,30,30]
            amp = jnp.array([0, 1, 1j, -1])
            sources = geometry.ComplexSources(positions=(x_pos, y_pos), amplitude=amp)
            ```
    """
    amplitude: Field
    omega: float
    domain: Domain

    def to_field(self, t=0.0):
        r"""Returns the complex field corresponding to the
        sources distribution.
        """
        return self.amplitude * jnp.exp(1j * self.omega * t)


@dataclass
class Sensors:
    """Sensors structure
    Attributes:
        positions (Tuple[List[int]]): sensors positions
    !!! example
        ```python
        x_pos = [10,20,30,40]
        y_pos = [30,30,30,30]
        sensors = geometry.Sensors(positions=(x_pos, y_pos))
        ```
    """

    positions: Tuple[jnp.ndarray]

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


@dataclass
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

    def to_array(self):
        r"""Returns the time-axis as an array"""
        return jnp.arange(0, self.t_end, self.dt)

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
        dt = cfl * min(medium.domain.dx) / jnp.max(medium.sound_speed)
        if t_end is None:
            t_end = jnp.sqrt(
                sum((x[-1] - x[0]) ** 2 for x in medium.domain.spatial_axis)
            ) / jnp.min(medium.sound_speed)
        return TimeAxis(dt=float(dt), t_end=float(t_end))


def _circ_mask(N, radius, centre):
    x, y = np.mgrid[0 : N[0], 0 : N[1]]
    dist_from_centre = np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2)
    mask = (dist_from_centre < radius).astype(int)
    return mask
