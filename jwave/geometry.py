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
    BACKWARD = 2
