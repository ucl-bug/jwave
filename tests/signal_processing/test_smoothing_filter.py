import numpy as np

from jwave.signal_processing import blackman, smoothing_filter


def test_smoothing_filter_with_1d_input():
    sample_input = np.ones(100)
    smooth_fun = smoothing_filter(sample_input)
    result = smooth_fun(sample_input)
    assert result.shape == sample_input.shape


def test_smoothing_filter_with_2d_input():
    sample_input = np.ones((10, 10))
    smooth_fun = smoothing_filter(sample_input)
    result = smooth_fun(sample_input)
    assert result.shape == sample_input.shape


def test_smoothing_filter_with_3d_input():
    sample_input = np.ones((10, 10, 10))
    smooth_fun = smoothing_filter(sample_input)
    result = smooth_fun(sample_input)
    assert result.shape == sample_input.shape


def test_smoothing_filter_with_complex_input():
    sample_input = (np.ones(100) + 1j * np.ones(100))
    smooth_fun = smoothing_filter(sample_input)
    result = smooth_fun(sample_input)
    assert np.all(np.isreal(result))    # check if output is real


def test_smoothing_filter_with_zero_input():
    sample_input = np.zeros(100)
    smooth_fun = smoothing_filter(sample_input)
    result = smooth_fun(sample_input)
    assert np.all(result == 0)


def test_blackman_function():
    result = blackman(100)
    assert len(result) == 100
