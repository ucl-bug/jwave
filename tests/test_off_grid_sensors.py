from jwave.geometry import _bli_function, BLISensors, Domain, FourierSeries
import numpy as np
import pytest


@pytest.mark.parametrize("n_grid", [(100, ), (101, )])
def test_bli_function(n_grid):
    # Make a load of sensors on the grid. Check the bli function.
    y = _bli_function(np.arange(n_grid), np.arange(n_grid), n_grid)
    # Assert that the bli function is 1 at one place for each detector.
    assert (np.all(np.sum(y != 0, axis=1) == 1))
    # Assert that the bli function is 0 everywhere else
    assert (np.all(np.sum(y == 0, axis=1) == n_grid - 1))
    # Assert that the 1 is in the correct place.
    assert (np.all(y[np.arange(n_grid), np.arange(n_grid)] == 1))

    # Check off-grid points:
    y = _bli_function(np.arange(0, n_grid - 1) + 0.25, np.arange(n_grid), n_grid)
    # Check that the sensor is non-zero at more than one place.
    assert (np.all(np.sum(y != 0, axis=1) > 1))
    assert (np.all(np.isclose(np.sum(y, axis=1), 1)))


@pytest.mark.parametrize("nx,ny,nz", [(100, 100, 100), (100, 101, 102)])
def test_sensor(nx, ny, nz):
    # Check that it is identical to on-grid detectors in that case.
    n_detectors = min(nx, ny, nz)

    xi = np.arange(n_detectors)
    yi = np.arange(n_detectors)
    zi = np.arange(n_detectors)
    np.random.shuffle(xi)
    np.random.shuffle(yi)
    np.random.shuffle(zi)

    x = xi.astype(float)
    y = yi.astype(float)
    z = zi.astype(float)

    p = np.random.random((nx, ny, nz))

    s1d = BLISensors((x,), (nx,))
    s2d = BLISensors((x, y), (nx, ny))
    s3d = BLISensors((x, y, z), (nx, ny, nz))

    domain1d = Domain((nx,), (1,))
    p1d = FourierSeries(p[:, 0, 0], domain1d)

    domain2d = Domain((nx, ny), (1, 1))
    p2d = FourierSeries(p[:, :, 0], domain2d)

    domain3d = Domain((nx, ny, nz), (1, 1, 1))
    p3d = FourierSeries(p, domain3d)

    result = s1d(p1d, None, None)
    assert (np.all(result[..., 0] == p[xi, 0, 0]))

    result = s2d(p2d, None, None)
    assert (np.all(result[..., 0] == p[xi, yi, 0]))

    result = s3d(p3d, None, None)
    assert (np.all(result[..., 0] == p[xi, yi, zi]))

    # Check off-grid (perturb a bit):
    s3d = BLISensors((x + 0.25, y + 0.3, z + 0.1), (nx, ny, nz))
    domain3d = Domain((nx, ny, nz), (1, 1, 1))
    # Check ones in ones out.
    p3d = FourierSeries(np.ones((nx, ny, nz)), domain3d)
    y = s3d(p3d, None, None)
    assert (np.all(np.isclose(y, 1)))

    # Check zeros in zeros out
    p3d = FourierSeries(np.zeros((nx, ny, nz)), domain3d)
    y = s3d(p3d, None, None)
    assert (np.all(y == 0))


if __name__ == "__main__":
    test_bli_function(100)
    test_bli_function(101)
    test_sensor()
    test_sensor(100, 101, 102)
