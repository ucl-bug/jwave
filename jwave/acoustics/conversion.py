import numpy as np
from jax import numpy as jnp

from jwave.geometry import Sensors


def pressure_from_density(
    sensors_data: jnp.ndarray, sound_speed: jnp.ndarray, sensors: Sensors
) -> jnp.ndarray:
  r"""
  Calculate pressure from acoustic density given by the raw output of the
  timestepping scheme.

  Args:
      sensors_data: Raw output of the timestepping scheme.
      sound_speed: Sound speed of the medium.
      sensors: Sensors object.

  Returns:
      jnp.ndarray: Pressure time traces at sensor locations
  """
  if sensors is None:
    return jnp.sum(sensors_data[1], -1) * (sound_speed ** 2)
  else:
    return jnp.sum(sensors_data[1], -1) * (sound_speed[sensors.positions] ** 2)

def db2neper(
  alpha: jnp.ndarray,
  y: jnp.ndarray,
):
  r'''
  Transforms absorption units from decibels to nepers.
  See http://www.k-wave.org/documentation/db2neper.php

  Args:
      alpha(jnp.ndarray): Absorption coefficient in decibels.
      y(jnp.ndarray): Distance in meters.

  Returns:
      jnp.ndarray: Absorption coefficient in nepers.
  '''
  return 100 * alpha * ((1e-6/(2*np.pi))**y) / (20 * np.log10(np.exp(1)))
