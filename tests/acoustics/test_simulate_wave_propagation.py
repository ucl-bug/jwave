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


if __name__ == "__main__":
    test_correct_call()
