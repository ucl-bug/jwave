import numpy as np

from jwave.geometry import (fibonacci_sphere, points_on_circle,
                            unit_fibonacci_sphere)


def testpoints_on_circle():
    n = 5
    radius = 10.0
    centre = (0.0, 0.0)
    x_expected = [10, 3, -8, -8, 3]
    y_expected = [0, 9, 5, -5, -9]

    x_actual, y_actual = points_on_circle(n, radius, centre, cast_int=True)

    assert x_actual == x_expected
    assert y_actual == y_expected


def testunit_fibonacci_sphere():
    samples = 128

    points = unit_fibonacci_sphere(samples=samples)

    # Assert that the correct number of points have been generated
    assert len(points) == samples

    # Assert that all points lie on the unit sphere
    for point in points:
        x, y, z = point
        distance_from_origin = np.sqrt(x**2 + y**2 + z**2)
        assert np.isclose(distance_from_origin, 1.0, atol=1e-5)


def testfibonacci_sphere():
    n = 128
    radius = 10.0
    centre = np.array([1.0, 2.0, 3.0])

    x, y, z = fibonacci_sphere(n, radius, centre, cast_int=False)

    # Assert that the correct number of points have been generated
    assert len(x) == len(y) == len(z) == n

    # Assert that all points lie on the sphere with the given radius and center
    for i in range(n):
        distance_from_centre = np.sqrt((x[i] - centre[0])**2 +
                                       (y[i] - centre[1])**2 +
                                       (z[i] - centre[2])**2)
        assert np.isclose(distance_from_centre, radius, atol=1e-5)


if __name__ == "__main__":
    test_repr()
    testpoints_on_circle()
    testunit_fibonacci_sphere()
    testfibonacci_sphere()
