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

from typing import Callable

from jax import numpy as jnp
from jaxdf import Continuous, Field, operator
from jaxdf.operators import compose

from jwave.geometry import Domain, Medium


def _base_pml(
    transform_fun: Callable,
    medium: Medium,
    exponent: float = 2.0,
    alpha_max: float = 2.0,
    shift=0.0,
) -> Field:

    def pml_edge(x):
        return x / 2 - medium.pml_size

    delta_pml = jnp.asarray(list(map(pml_edge, medium.domain.N)))
    coord_grid = Domain(N=medium.domain.N,
                        dx=tuple([1.0] * len(medium.domain.N))).grid
    coord_grid = coord_grid + shift

    def _pml_fun(x, delta_pml):
        diff = (jnp.abs(x) - 1.0 * delta_pml) / medium.pml_size
        on_pml = jnp.where(diff > 0, diff, 0)
        alpha = alpha_max * (on_pml**exponent)
        exp_val = transform_fun(alpha)
        return exp_val

    return _pml_fun(coord_grid, delta_pml)


def complex_pml_on_grid(medium: Medium,
                        omega: float,
                        exponent=4.0,
                        alpha_max=2.0,
                        shift=0.0) -> jnp.ndarray:
    transform_fun = lambda alpha: 1.0 / (1 + 1j * alpha)
    return _base_pml(transform_fun, medium, exponent, alpha_max, shift=shift)


def td_pml_on_grid(
    medium: Medium,
    dt: float,
    exponent=4.0,
    alpha_max=2.0,
    c0=1.0,
    dx=1.0,
    coord_shift=0.0,
) -> jnp.ndarray:

    if medium.domain.ndim not in [1, 2, 3]:
        raise NotImplementedError(
            f"Can't make a PML for a domain of dimensions {medium.domain.ndim}"
        )

    if medium.pml_size == 0:
        size = tuple(list(medium.domain.N) + [1])
        return jnp.ones(size)

    x_right = (jnp.arange(1, medium.pml_size + 1, 1) +
               coord_shift) / medium.pml_size
    x_left = (jnp.arange(medium.pml_size, 0, -1) -
              coord_shift) / medium.pml_size
    x_right = x_right**exponent
    x_left = x_left**exponent

    alpha_left = jnp.exp(alpha_max * (-1) * x_left * dt * c0 / 2 / dx)
    alpha_right = jnp.exp(alpha_max * (-1) * x_right * dt * c0 / 2 / dx)

    pml_shape = tuple(list(medium.domain.N) + [len(medium.domain.N)])
    pml = jnp.ones(pml_shape)

    if medium.domain.ndim >= 1:
        pml = pml.at[..., :medium.int_pml_size, -1].set(alpha_left)
        pml = pml.at[..., -medium.int_pml_size:, -1].set(alpha_right)

    if medium.domain.ndim >= 2:
        alpha_left = jnp.expand_dims(alpha_left, -1)
        alpha_right = jnp.expand_dims(alpha_right, -1)
        pml = pml.at[..., :medium.int_pml_size, :, -2].set(alpha_left)
        pml = pml.at[..., -medium.int_pml_size:, :, -2].set(alpha_right)

    if medium.domain.ndim == 3:
        alpha_left = jnp.expand_dims(alpha_left, -1)
        alpha_right = jnp.expand_dims(alpha_right, -1)
        pml = pml.at[:medium.int_pml_size, :, :, -3].set(alpha_left)
        pml = pml.at[-medium.int_pml_size:, :, :, -3].set(alpha_right)

    return pml


def _sigma(x):
    alpha = 2.0
    sigma_star = 10.0
    delta_pml = 54.0
    L_half = 64.0

    abs_x = jnp.abs(x)
    in_pml_amplitude = (jnp.abs(abs_x - delta_pml) /
                        (L_half - delta_pml))**alpha
    return jnp.where(abs_x > delta_pml, sigma_star * in_pml_amplitude, 0.0)


@operator
def complex_pml(x: Continuous,
                medium: Medium,
                *,
                omega=1.0,
                sigma_star=10.0,
                alpha=2.0,
                params=None):
    dx = x.domain.dx
    N = x.domain.N

    def sigma(x):
        delta_pml = dx[0] * (N[0] / 2 - medium.pml_size)
        L_half = dx[0] * N[0] / 2
        abs_x = jnp.abs(x)
        in_pml_amplitude = (jnp.abs(abs_x - delta_pml) /
                            (L_half - delta_pml))**alpha
        return jnp.where(abs_x > delta_pml, sigma_star * in_pml_amplitude, 0.0)

    y = compose(x)(sigma)
    return 1.0 / (1.0 + 1j * y / omega)
