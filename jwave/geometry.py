from jax import numpy as jnp
from functools import reduce
from typing import NamedTuple, Tuple
from enum import IntEnum


class Domain(NamedTuple):
    r"""A rectangular domain."""
    N: Tuple[int]
    dx: Tuple[float]

    @property
    def size(self):
        r"""Returns the lenght of the grid sides

        !!! example
            ```python
            L = grid.domain_size
            ```

        """
        return list(map(lambda x, y: x * y, zip(self.N, self.dx)))

    @property
    def ndim(self):
        return len(self.N)

    @property
    def cell_volume(self):
        return reduce(lambda x, y: x * y, self.dx)

    @property
    def spatial_axis(self):
        def _make_axis(n, delta):
            if n % 2 == 0:
                return jnp.arange(0, n) * delta - delta * n / 2
            else:
                return jnp.arange(0, n) * delta - delta * (n - 1) / 2

        axis = [_make_axis(n, delta) for n, delta in zip(self.N, self.dx)]
        axis = [ax - jnp.mean(ax) for ax in axis]
        return axis

    @property
    def origin(self):
        return jnp.zeros((self.ndim,))

    @staticmethod
    def _make_grid_from_axis(axis):
        return jnp.stack(jnp.meshgrid(*axis, indexing="ij"), axis=-1)

    @property
    def grid(self):
        """Returns a grid of spatial position, of size
        `Nx x Ny x Nz x ... x num_axis` such that the element
        `[x1,x2,x3, .., :]` is a coordinate vector.
        """
        axis = self.spatial_axis
        return self._make_grid_from_axis(axis)


class Staggered(IntEnum):
    r"""Staggering flags as enumerated constants. This makes sure
    that we are consistent when asking staggered computations
    across different spectral functions

    Attributes:
        NONE: Unstaggered
        FORWARD: Staggered forward
        BACKWARD: Staggered backward
    """
    NONE = 0
    FORWARD = 1
    BACKWARD = -1


class Medium(NamedTuple):
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
            density = jnp.ones(N),
            attenuation = 0.0,
            pml_size = 15
        )
        ```
    """
    domain: Domain
    sound_speed: jnp.ndarray
    density: jnp.ndarray
    attenuation: jnp.ndarray
    pml_size: int


class Sources(NamedTuple):
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


class ComplexSources(NamedTuple):
    r"""ComplexSources structure
    Attributes:
        positions (Tuple[List[int]): source positions
        amplitude (jnp.ndarray): source complex amplitudes
    !!! example
        ```python
        x_pos = [10,20,30,40]
        y_pos = [30,30,30,30]
        amp = jnp.array([0, 1, 1j, -1])
        sources = geometry.ComplexSources(positions=(x_pos, y_pos), amplitude=amp)
        ```
    """
    positions: Tuple[jnp.ndarray]
    amplitude: Tuple[jnp.ndarray]

    def to_field(self, grid):
        r"""Returns the complex field corresponding to the
        sources distribution.
        """
        field = jnp.zeros(grid.N, dtype=jnp.complex64)
        field = field.at[self.positions].set(self.amplitude)
        return field


class Sensors(NamedTuple):
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


class TimeAxis(NamedTuple):
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
    def from_kgrid(medium: Medium, cfl: float = 0.3, t_end=None):
        r"""Construct a `TimeAxis` object from `kGrid` and `Medium`
        Args:
            grid (kGrid):
            medium (Medium):
            cfl (float, optional):  The [CFL number](http://www.k-wave.org/). Defaults to 0.3.
            t_end ([float], optional):  The final simulation time. If None,
                    it is automatically calculated as the time required to travel
                    from one corner of the domain to the opposite one.
        """
        dt = dt = cfl * min(medium.domain.dx) / jnp.max(medium.sound_speed)
        if t_end is None:
            t_end = jnp.sqrt(
                sum((x[-1] - x[0]) ** 2 for x in medium.domain.spatial_axis)
            ) / jnp.min(medium.sound_speed)
        return TimeAxis(dt=float(dt), t_end=float(t_end))