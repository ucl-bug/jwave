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

from jax.lax import scan
from jax import jit
from jwave.transformations import ScanCheckpoint, CheckpointType
from pytest import mark
from jax import random
from jax import numpy as jnp

KEY = random.PRNGKey(42)

# Check they give the same results as jax.lax.scan
# Check that the function is jittable

@mark.parametrize(
    "checkpoint", [
        CheckpointType.NONE,
    ])
def test_scan_equivalente(checkpoint_type):

    def scan_fun(carry, x):
        return carry + x, carry + 2*x
    
    init = 0
    xs = jnp.random.uniform(KEY, (20,))
    scan_checkpoint = ScanCheckpoint(checkpoint_type)

    # Scan with jax
    out_carry, y = scan(scan_fun, init, xs)

    # Scan with jwave
    j_out_carry, j_y = scan_checkpoint.scan(scan_fun, init, xs)

    assert jnp.allclose(out_carry, j_out_carry)
    assert jnp.allclose(y, j_y)