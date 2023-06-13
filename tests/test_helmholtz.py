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

from jax import numpy as jnp

from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave.geometry import Domain, FourierSeries, Medium


def test_if_homog_helmholtz_runs():
    N = (128, 128)
    domain = Domain(N, (1.0, 1.0))
    src_field = jnp.zeros(N).astype(jnp.complex64)
    src_field = src_field.at[64, 22].set(1.0)
    src_field = jnp.expand_dims(src_field, axis=-1)
    src_field = FourierSeries(src_field, domain)

    medium = Medium(domain, sound_speed=1.0, pml_size=15)

    field = helmholtz_solver(
        medium,
        1.0,
        src_field,
        tol=1e-5,
        restart=5,
        method="gmres",
        maxiter=10,
    )


if __name__ == "__main__":
    test_if_homog_helmholtz_runs()
