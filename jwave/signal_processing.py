from typing import Callable

from jax import eval_shape
from jax import numpy as jnp
from jax import vmap


def analytic_signal(x, axis=-1):
    """
    Computes the analytic signal from a real signal `x`, using the
    FFT.

    Args:
        x (jnp.ndarray): [description]
        axis (int, optional): [description]. Defaults to -1.

    Returns:
        jnp.ndarray: [description]
    """
    spectrum = jnp.fft.fft(x, axis=axis)

    # Set negative frequencies to zero along the axis, using slices
    positive_indices = slice(0, spectrum.shape[axis] // 2)
    slices = [slice(None)] * spectrum.ndim
    slices[axis] = positive_indices
    slices = tuple(slices)
    spectrum = spectrum.at[slices].set(0.0)

    # Get complex signal
    x = jnp.fft.ifft(spectrum, axis=axis)
    return x


def fourier_downsample(x, subsample=2, discard_last=True):
    """
    Downsample a signal by taking the Fourier transform and
    discarding the high frequencies.

    Args:
        x (jnp.ndarray): Signal to be downsampled
        subsample (int, optional): Sumsampling factor. Defaults to 2.
        discard_last (bool, optional): If True, the last dimension is not
            subsampled. Defaults to True.

    Returns:
        jnp.ndarray: [description]
    """
    if subsample == 1:
        return x

    def _single_downsample(x):
        """removes positive and negative frequency at appropriate
        cut values"""
        Fx = jnp.fft.fftshift(jnp.fft.fftn(x))
        cuts = [int((subsample - 1) * x / 2 / subsample) for x in Fx.shape]
        slices = tuple([slice(cut, -cut) for cut in cuts])
        return jnp.fft.ifftn(jnp.fft.ifftshift(Fx[slices])) / (subsample ** x.ndim)

    if discard_last:
        _single_downsample = vmap(_single_downsample, in_axes=(-1,), out_axes=-1)

    return _single_downsample(x)


def fourier_upsample(x, upsample=2, discard_last=True):
    """
    Upsample a signal by taking the Fourier transform and
    adding zeros at the high frequencies.

    Args:
        x (jnp.ndarray): Signal to be upsampled
        upsample (int, optional): Upsampling factor. Defaults to 2.

    Returns:
        jnp.ndarray: Upsampled signal
    """
    if upsample == 1:
        return x

    def _single_upsample(x):
        """adds zeros at appropriate cut values"""
        new_size = list(map(lambda x: x * upsample, x.shape))
        Fx = jnp.fft.fftshift(jnp.fft.fftn(x))
        new_Fx = jnp.zeros(new_size, dtype=Fx.dtype)
        cuts = [int((upsample - 1) * x / 2 / upsample) for x in new_size]
        slices = tuple([slice(cut, -cut) for cut in cuts])
        new_Fx = new_Fx.at[slices].set(Fx)
        return jnp.fft.ifftn(jnp.fft.ifftshift(new_Fx)) * (upsample ** x.ndim)

    if discard_last:
        _single_upsample = vmap(_single_upsample, in_axes=(-1,), out_axes=-1)

    return _single_upsample(x)


def apply_ramp(
    signal: jnp.ndarray, dt: float, center_freq: float, warmup_cycles: float = 3
) -> jnp.ndarray:
    r"""Processes the signal $`s(t)`$ as

    ```math
    s(t)\cdot \text{min}(1, f_0t/\sigma)
    ```

    Args:
        signal (jnp.ndarray): [description]
        dt (float): [description]
        center_freq (float): $`f_0`$
        warmup_cycles (float, optional): $`\sigma`$. Defaults to 3.

    Returns:
        jnp.ndarray: [description]
    """

    t = jnp.arange(signal.shape[0]) * dt
    period = 1 / center_freq
    ramp_length = warmup_cycles * period
    return signal * jnp.where(t < ramp_length, (t / ramp_length), 1.0)


def blackman(N: int):
    r"""Returns the blackman window of length `N`

    Args:
        N (int): [description]

    Returns:
        [type]: [description]
    """
    i = jnp.arange(N)
    return 0.42 - 0.5 * jnp.cos(2 * jnp.pi * i / N) + 0.08 * jnp.cos(4 * jnp.pi * i / N)


def gaussian_window(
    signal: jnp.ndarray, time: jnp.ndarray, mu: float, sigma: float
) -> jnp.ndarray:
    r"""Returns the gaussian window

    ```math
    s(t)\cdot \exp \left( - \frac{(t-\mu)^2}{\sigma^2} \right)
    ```

    Args:
        signal (jnp.ndarray): $`s(t)`$
        time (jnp.ndarray): $`t`$
        mu (float): $`\mu`$
        sigma (float): $`\sigma`$

    Returns:
        jnp.ndarray: [description]
    """
    return signal * jnp.exp(-((time - mu) ** 2) / sigma ** 2)


def smoothing_filter(sample_input) -> Callable:
    r"""Returns a smoothing filter based on the blackman window"""
    # Constructs the filter
    dimensions = sample_input.shape
    axis = [blackman(x) for x in dimensions]
    if len(dimensions) == 1:
        filter_kernel = jnp.fft.fftshift(axis[0])
    else:
        # TODO: Find a more elegant way of constructing the filter
        if len(axis) == 1:
            filter_kernel = axis[0]
        elif len(axis) == 2:
            filter_kernel = jnp.fft.fftshift(jnp.outer(*axis))
        elif len(axis) == 3:
            filter_kernel_2d = jnp.outer(*axis[1:])
            third_component = jnp.expand_dims(jnp.expand_dims(axis[0], 1), 2)
            filter_kernel = third_component * filter_kernel_2d

    # Different filtering functions for real and complex data
    if sample_input.dtype != jnp.complex64 or sample_input.dtype != jnp.complex128:
        Fx = eval_shape(jnp.fft.rfft, sample_input)
        filter_kernel = filter_kernel[..., : Fx.shape[-1]]

        def smooth_fun(x):
            return jnp.fft.irfftn(filter_kernel * jnp.fft.rfftn(x)).real

    else:

        def smooth_fun(x):
            return jnp.fft.ifftn(filter_kernel * jnp.fft.fftn(x)).real

    return smooth_fun


def smooth(
  x: jnp.ndarray,
  exponent: float = 1.0,
) -> jnp.ndarray:
    """Smooths a  n-dimensioanl signal by multiplying its
    spectrum by a blackman window.

    Args:
        x (jnp.ndarray): [description]

    Returns:
        jnp.ndarray: [description]
    """
    dimensions = x.shape
    axis = [blackman(x) for x in dimensions]
    if len(dimensions) == 1:
        filter_kernel = jnp.fft.fftshift(axis[0])
    else:
        # TODO: Find a more elegant way of constructing the filter
        if len(axis) == 1:
            filter_kernel = jnp.fft.fftshift(axis[0])
        elif len(axis) == 2:
            filter_kernel = jnp.fft.fftshift(jnp.outer(*axis))
        elif len(axis) == 3:
            filter_kernel_2d = jnp.outer(*axis[1:])
            third_component = jnp.expand_dims(jnp.expand_dims(axis[0], 1), 2)
            filter_kernel = jnp.fft.fftshift(third_component * filter_kernel_2d)
    filter_kernel = filter_kernel**exponent
    return jnp.fft.ifftn(filter_kernel * jnp.fft.fftn(x)).real


def _dist_from_ends(N):
    return jnp.concatenate([jnp.arange(N // 2), jnp.flip(jnp.arange(0, N - N // 2))])


def tone_burst(
    sample_freq: float, signal_freq: float, num_cycles: float
) -> jnp.ndarray:
    r"""Returns a tone burst

    Args:
        sample_freq (float): Sampling frequency
        signal_freq (float): Signal frequency
        num_cycles (float): Number of cycles

    Returns:
        jnp.ndarray: The tone burst signal
    """

    def gaussian(x, magnitude, mean, variance):
        return magnitude * jnp.exp(-((x - mean) ** 2) / (2 * variance))

    dt = 1 / sample_freq
    tone_length = num_cycles / signal_freq
    tone_t = jnp.arange(0, tone_length + dt, dt)
    tone_burst = jnp.sin(2 * jnp.pi * signal_freq * tone_t)

    # Gaussian window
    x_lim = 3
    window_x = jnp.linspace(-x_lim, x_lim, tone_burst.shape[0])
    window = gaussian(window_x, 1, 0, 1)
    return tone_burst * window
