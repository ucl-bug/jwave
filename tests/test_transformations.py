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
from jax import random
from jax.lax import scan
from pytest import mark

from jwave.transformations import CheckpointType, ScanCheckpoint

KEY = random.PRNGKey(42)

# Check they give the same results as jax.lax.scan
# Check that the function is jittable


@mark.parametrize(
    "checkpoint_type",
    [
        CheckpointType.NONE,
        CheckpointType.STEP,
        CheckpointType.TREEVERSE,
        CheckpointType.DIVIDE_AND_CONQUER,
    ],
)
def test_scan_equivalent(checkpoint_type):

    def scan_fun(carry, x):
        return carry + x, carry + 2 * x

    init = 0
    xs = random.uniform(KEY, (20, ))
    scan_checkpoint = ScanCheckpoint(checkpoint_type)

    # Scan with jax
    out_carry, y = scan(scan_fun, init, xs)

    # Scan with jwave
    j_out_carry, j_y = scan_checkpoint(scan_fun, init, xs)

    assert jnp.allclose(out_carry, j_out_carry)
    assert jnp.allclose(y, j_y)
