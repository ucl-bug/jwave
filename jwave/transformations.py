import jax
from jax.lax import scan
from typing import Callable, Tuple, Union, TypeVar, Sequence
from dataclasses import Enum, dataclass
from jax import numpy as jnp

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')

class ScanCheckpoint(object):
    def __init__(
        self,
        kind = "None",
        max_length: int = 4,
    ):
        self.kind = kind
        self.max_length = max_length

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, value: str):
        known_kinds = ["None", "Step", "Treeverse", "DivideAndConquer"]
        assert value in known_kinds, f"ScanCheckpoint.kind must be in {known_kinds}"
        self._kind = value

def step_checkpoint_scan(
  f: Callable[[Carry, X], Tuple[Carry, Y]],
  init: Carry,
  xs: X,
):
  f = jax.checkpoint(f)
  return jax.lax.scan(scan_fun, fields, output_steps)

def dvide_and_conquer_scan(
  f: Callable[[Carry, X], Tuple[Carry, Y]],
  init: Carry,
  xs: X,
  max_length = 4,
):

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
    
  return dec_operation(init, x)

def treeverse_scan(
  f: Callable[[Carry, X], Tuple[Carry, Y]],
  init: Carry,
  lengths: Union[None, Sequence[int]]
) -> Tuple[Carry, Y]:
  r"""
  Performs a scan operation with treeverse checkpointing.

  Args:
    f: A function that takes a carry and an input and returns a new carry and an output.
    init: The initial carry.
    lengths: A sequence of integers specifying the lengths of the sequences to be run with 
        a single scan call. Checkpointing is performed after each sequence.

  Returns:
    A tuple of the final carry and the final output.
  """
  @jax.named_call
  def inner_scan_fun(carry, x):
    return jax.lax.scan(f, carry, jnp.arange(x))
  
  @jax.jit
  def scanned_fun(init):
    y = None
    carry = init
    for length in lengths:
      carry, y_extra = jax.checkpoint(inner_scan_fun, static_argnums=(1,))(carry, length)
      
      # Append the
      if y is None:
        y = y_extra
      else:
        y = jax.tree_map(lambda x, y: jnp.concatenate([x, y]), y, y_extra)
    return carry, y
  
  return scanned_fun