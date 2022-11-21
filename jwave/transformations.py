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

import jax
from jax.lax import scan
from typing import Callable, Tuple, Union, TypeVar, Sequence
from enum import Enum
from jax import numpy as jnp
from functools import partial

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')

class CheckpointType(Enum):
  NONE = 0
  STEP = 1
  TREEVERSE: 2
  DIVIDE_AND_CONQUER = 3

class ScanCheckpoint(object):
    def __init__(
        self,
        kind: CheckpointType = CheckpointType.NONE,
        max_length: int = 4,
    ):
        self.kind = kind
        self.max_length = max_length

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, value: CheckpointType):
      self._kind = value

    @staticmethod
    def no_checkpoint_scan(
      f: Callable[[Carry, X], Tuple[Carry, Y]],
      init: Carry,
      xs: Sequence[X],
    ) -> Tuple[Carry, Sequence[Y]]:
      r"""Lifted version of scan that does not checkpoint. 
      Equivalent to `jax.lax.scan(f, init, xs)`.
      
      Args:
        f (Callable): A function from (carry, x) to (carry, y).
        init (Carry): The initial carry value.
        xs (Sequence[X]): The sequence of values to be scanned over.

      Returns:
        A pair (final_carry, ys) where `ys[i] = f(...f(f(init, xs[0]), xs[1])..., xs[i])`.
      """
      return scan(f, init, xs)

    @staticmethod
    def step_checkpoint_scan(
      f: Callable[[Carry, X], Tuple[Carry, Y]],
      init: Carry,
      xs: X,
    ) -> Tuple[Carry, Sequence[Y]]:
      r"""Equivalent to `jax.lax.scan(f, init, xs)` but with checkpointing
      after each step. This means that no intermediate results computed
      by `f` are saved, but only the intermediate results of the scan```

      Args:
        f (Callable): A function from (carry, x) to (carry, y).
        init (Carry): The initial carry value.
        xs (Sequence[X]): The sequence of values to be scanned over.

      Returns:
        A pair (final_carry, ys) where `ys[i] = f(...f(f(init, xs[0]), xs[1])..., xs[i])`.
      """
      f = jax.checkpoint(f)
      return jax.lax.scan(f, init, xs)

    @staticmethod
    def dvide_and_conquer_scan(
      f: Callable[[Carry, X], Tuple[Carry, Y]],
      init: Carry,
      xs: X,
      max_length: int = 4,
    ) -> Tuple[Carry, Sequence[Y]]:
      r"""Equivalent to `jax.lax.scan(f, init, xs)` but with recursive checkpointing, see
      [[Siskind and Pearlmutter, 2017](https://arxiv.org/abs/1708.06799)]. The scan is
      recursively split into two halves until the length of the sequence is less than
      `max_length`. Each half is then checkpointed and the results are concatenated.
      This results in a space and time complexity of `O(log(N/M))` where `N` is the length
      of the sequence and `M` is the maximum length of each subsequence.

      Args:
        f (Callable): A function from (carry, x) to (carry, y).
        init (Carry): The initial carry value.
        xs (Sequence[X]): The sequence of values to be scanned over.
        max_length (int): The maximum length of each subsequence. 
      
      Returns:
        A pair (final_carry, ys) where `ys[i] = f(...f(f(init, xs[0]), xs[1])..., xs[i])`.
      """

      @jax.jit
      def dec_operation(carry, x: X):
        length = x.shape[0]
        if length <= max_length:
          return jax.lax.scan(f, carry, x)
        else:
          # Split the x
          x_1 = x[:length//2]
          x_2 = x[length//2:]

          # Run each in sequence, with appropriate carry
          carry, y_1 = jax.checkpoint(dec_operation)(carry, x_1)
          carry, y_2 = jax.checkpoint(dec_operation)(carry, x_2)

          # Concatenate results
          y = jax.tree_map(lambda x, y: jnp.concatenate([x, y]), y_1, y_2)
          return carry, y
        
      return dec_operation(init, xs)

    @staticmethod
    def treeverse_scan(
      f: Callable[[Carry, X], Tuple[Carry, Y]],
      init: Carry,
      xs: X,
      max_length: int = 4,
    ) -> Tuple[Carry, Sequence[Y]]:
      r"""Equivalent to `jax.lax.scan(f, init, xs)` but checkpoints every
      `max_length` steps. This means that intermediate results computed
      by `f` are saved every `max_length` steps. This results in a space
      complexity of `O(N/M)` and a time complexity of `O(N*M)` where `N` is
      the length of the sequence and `M` is the maximum length of each subsequence.

      Args:
        f (Callable): A function from (carry, x) to (carry, y).
        init (Carry): The initial carry value.
        xs (Sequence[X]): The sequence of values to be scanned over.
        max_length (int): The maximum length of each subsequence.

      Returns:
        A pair (final_carry, ys) where `ys[i] = f(...f(f(init, xs[0]), xs[1])..., xs[i])`.
      """

      @jax.named_call
      def inner_scan_fun(carry, x):
        return jax.lax.scan(f, carry, jnp.arange(x))

      # Split sequence
      x_splitted = [xs[i:i + max_length] for i in range(0, len(xs), max_length)]
      
      @jax.jit
      def scanned_fun(init):
        y = None
        carry = init
        for this_x in x_splitted:
          carry, y_extra = jax.checkpoint(inner_scan_fun, static_argnums=(1,))(carry, this_x)
          
          # Append the
          if y is None:
            y = y_extra
          else:
            y = jax.tree_map(lambda x, y: jnp.concatenate([x, y]), y, y_extra)
        return carry, y
      
      return scanned_fun(init)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
      self, 
      f: Callable[[Carry, X], Tuple[Carry, Y]],
      init: Carry,
      xs: X,
    ) -> Tuple[Carry, Sequence[Y]]:
      r"""Equivalent to `jax.lax.scan(f, init, xs)` but with checkpointing
      depending on the `self.kind` and `self.max_length` attributes.
      This function can be transformed with jax function transformations (
      `jit`, `grad`, etc...)

      Args:
        f (Callable): A function from (carry, x) to (carry, y).
        init (Carry): The initial carry value.
        xs (Sequence[X]): The sequence of values to be scanned over.

      Returns:
        A pair (final_carry, ys) where `ys[i] = f(...f(f(init, xs[0]), xs[1])..., xs[i])`.
      """
      func_map = {
        CheckpointType.NONE: self.no_checkpoint_scan,
        CheckpointType.STEP: self.step_checkpoint_scan,
        CheckpointType.DIVIDE_AND_CONQUER: self.dvide_and_conquer_scan,
        CheckpointType.TREEVERSE: self.treeverse_scan,
      }
      return func_map[self.checkpoint_type](f, init, xs)
      