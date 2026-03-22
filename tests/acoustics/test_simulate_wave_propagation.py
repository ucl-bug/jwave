import logging
from io import StringIO

from jax import numpy as jnp

from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Domain, FourierSeries, Medium, TimeAxis
from jwave.logger import logger, set_logging_level


def test_correct_call():
    domain = Domain((100, ), (1., ))
    fs = FourierSeries(jnp.ones((100, )), domain)
    medium = Medium(domain, sound_speed=fs)
    p0 = FourierSeries(jnp.zeros((100, )), domain)
    tax = TimeAxis.from_medium(medium)
    tax.t_end = 1.0

    # Create a StringIO object to capture log output
    log_capture_string = StringIO()
    ch = logging.StreamHandler(log_capture_string)

    # Add the custom handler to the logger
    logger.addHandler(ch)
    set_logging_level(logging.DEBUG)

    # Run the function
    p = simulate_wave_propagation(medium, tax, p0=p0)

    # Remove the handler after capturing the logs
    logger.removeHandler(ch)

    # Get the log output from the StringIO object
    log_contents = log_capture_string.getvalue()

    # Restore logging level
    set_logging_level(logging.INFO)

    # Perform assertions on the log contents
    assert "Starting simulation using FourierSeries code" in log_contents


def test_fd_nondefault_accuracy():
    """Regression test for jwave#224: FD fields with accuracy != 8
    must not cause pytree mismatch in lax.scan."""
    from jwave import FiniteDifferences
    from jwave.acoustics import TimeWavePropagationSettings
    from jwave.geometry import circ_mask

    domain = Domain((64, 64), (1e-3, 1e-3))
    p0_arr = 5.0 * circ_mask(domain.N, 3, (32, 32))
    p0 = FiniteDifferences(
        jnp.expand_dims(p0_arr, -1), domain, accuracy=4)
    sound_speed = FiniteDifferences(
        jnp.expand_dims(jnp.ones(domain.N) * 1500.0, -1), domain, accuracy=4)
    medium = Medium(domain, sound_speed=sound_speed, pml_size=0)
    time_axis = TimeAxis.from_medium(medium, cfl=0.1)
    time_axis.t_end = 2e-6
    settings = TimeWavePropagationSettings(smooth_initial=False)

    p = simulate_wave_propagation(medium, time_axis, p0=p0, settings=settings)
    assert p is not None


if __name__ == "__main__":
    test_correct_call()
