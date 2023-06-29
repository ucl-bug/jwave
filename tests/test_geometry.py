import numpy as np
from jax import numpy as jnp

from jwave.geometry import (Domain, Medium, _fibonacci_sphere,
                            _unit_fibonacci_sphere, points_on_circle)


def test_repr():
    # Create Domain object. Replace with correct constructor based on your implementation.
    domain = Domain()

    N = (8, 9)
    medium = Medium(domain=domain,
                    sound_speed=jnp.ones(N),
                    density=jnp.ones(N),
                    attenuation=0.0,
                    pml_size=15)

    expected_output = "Medium:\n - domain: {}\n - sound_speed: {}\n - density: {}\n - attenuation: {}\n - pml_size: {}".format(
        str(medium.domain), str(medium.sound_speed), str(medium.density),
        str(medium.attenuation), str(medium.pml_size))

    # Check that the __repr__ method output matches the expected output
    assert str(medium) == expected_output


def testpoints_on_circle():
    n = 5
    radius = 10.0
    centre = (0.0, 0.0)
    x_expected = [10, 3, -8, -8, 3]
    y_expected = [0, 9, 5, -5, -9]

    x_actual, y_actual = points_on_circle(n, radius, centre, cast_int=True)

    assert x_actual == x_expected
    assert y_actual == y_expected


def test_unit_fibonacci_sphere():
    samples = 128

    points = _unit_fibonacci_sphere(samples=samples)

    # Assert that the correct number of points have been generated
    assert len(points) == samples

    # Assert that all points lie on the unit sphere
    for point in points:
        x, y, z = point
        distance_from_origin = np.sqrt(x**2 + y**2 + z**2)
        assert np.isclose(distance_from_origin, 1.0, atol=1e-5)


def test_fibonacci_sphere():
    n = 128
    radius = 10.0
    centre = np.array([1.0, 2.0, 3.0])

    x, y, z = _fibonacci_sphere(n, radius, centre, cast_int=False)

    # Assert that the correct number of points have been generated
    assert len(x) == len(y) == len(z) == n

    # Assert that all points lie on the sphere with the given radius and center
    for i in range(n):
        distance_from_centre = np.sqrt((x[i] - centre[0])**2 +
                                       (y[i] - centre[1])**2 +
                                       (z[i] - centre[2])**2)
        assert np.isclose(distance_from_centre, radius, atol=1e-5)
