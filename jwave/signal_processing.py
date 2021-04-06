from jax import numpy as jnp


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


def smooth(x: jnp.ndarray) -> jnp.ndarray:
    """Smooths a  n-dimensioanl signal by multiplying its
    spectrum by a blackman window.

    Args:
        x (jnp.ndarray): [description]

    Returns:
        jnp.ndarray: [description]
    """
    dimensions = x.shape
    axis = [blackman(x) for x in dimensions]
    filter = jnp.fft.fftshift(jnp.outer(*axis))
    return jnp.fft.ifftn(filter * jnp.fft.fftn(x)).real


def _dist_from_ends(N):
    return jnp.concatenate([jnp.arange(N // 2), jnp.flip(jnp.arange(0, N - N // 2))])


def tone_burst(sample_freq, signal_freq, num_cycles):
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
