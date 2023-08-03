from typing import Optional

import numpy as np
from diffrax.custom_types import Scalar
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxdf.operators import functional

from jwave.geometry import Medium


@register_pytree_node_class
class TimeAxis:
    r"""Temporal vector to be used for ODE integration

    Args:
      t0 (Scalar): Initial time
      t1 (Scalar): Final time
      dt (Optional[Scalar], optional): Time step. If None, it is automatically
        calculated by the solver. Defaults to None.

    """
    t0: Scalar
    t1: Scalar
    dt: Optional[Scalar] = None

    def __init__(self, t0: Scalar, t1: Scalar, dt: Optional[Scalar] = None):
        self.t0 = t0
        self.t1 = t1
        self.dt = dt

    def __repr__(self):
        return f"TimeAxis(t0={self.t0}, t1={self.t1}, dt={self.dt})"

    def __str__(self):
        return self.__repr__()

    def tree_flatten(self):
        children = (
            self.t0,
            self.t1,
            self.dt,
        )
        aux = None
        return (children, aux)

    @classmethod
    def tree_unflatten(cls, aux, children):
        t0, t1, dt = children
        return cls(t0, t1, dt)

    @property
    def Nt(self):
        r"""Returns the number of time steps"""
        Δt = self.t1 - self.t0
        return np.ceil(Δt / self.dt).astype(int)

    def to_array(self, keep_last: bool = False):
        r"""Returns the time-axis as an array"""
        if keep_last:
            out_steps = jnp.linspace(self.t0, self.t1, self.Nt + 1)
        else:
            out_steps = jnp.linspace(self.t0, self.t1 - self.dt, self.Nt)
        return out_steps

    @classmethod
    def from_cfl_number(
        cls,
        medium: Medium,
        cfl: Scalar = 0.3,
        t0: Scalar = 0.0,
        t1: Optional[Scalar] = None,
    ) -> "TimeAxis":
        r"""Construct a `TimeAxis` object from a medium, for a given
      [Courant-Friedrichs-Lewy (CFL) number](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition).

      Args:
        medium (Medium): Medium object
        cfl (Scalar, optional): CFL number. Defaults to 0.3.
        t0 (Scalar, optional): Initial time. Defaults to 0.0.
        t1 (Optional[Scalar], optional): Final time. If None, it is automatically
          computer as the time it takes for a wave to travel from one end of the
          domain to the other, under the smallest sound speed. Defaults to None.

      Returns:
        TimeAxis: TimeAxis object
      """
        dt = cfl * min(medium.domain.dx) / functional(medium.sound_speed)(
            np.max)
        if t1 is None:
            t1 = np.sqrt(
                sum((x[-1] - x[0])**2
                    for x in medium.domain.spatial_axis)) / functional(
                        medium.sound_speed)(np.min)
        return cls(t0, t1, dt)
