from jax import numpy as jnp
from jax import eval_shape
from typing import Tuple, Union
from jwave.geometry import Domain

def rfft_interp(
    k: jnp.ndarray, 
    spectrum: jnp.ndarray, 
    x: jnp.ndarray, 
    first_dim_size: int
) -> float:
    r"""Calculates the value of a field $`f`$ at an arbitrary position
    $`x`$ using 

    ```math
    f(x) = \sum_i \hat f e^{ik^\top x}
    ```

    for real signals. The `spectrum` dimension and content, together
    with the frequency coordinate grid $`k`$ must conform to 
    [`jax.numpy.fft.rfftn`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.rfftn.html).

    Args:
        k (jnp.ndarray): [description]
        spectrum (jnp.ndarray): [description]
        x (jnp.ndarray): [description]
        first_dim_size (int): [description]

    Returns:
        The field value
    """    

    phase = jnp.exp(1j*jnp.sum(k*x,axis=-1))
    first_half = jnp.sum(spectrum*phase)

    # Compensate for missing frequency components due to 
    # conjugate simmetry of spectrum
    end_val = int(first_dim_size % 2 == 1)
    k_missing = k[...,1:-end_val,:]
    spectrum_missing = jnp.conj(spectrum[...,1:-end_val])
    phase = jnp.exp(-1j*jnp.sum(k_missing*x,axis=-1)) # TODO: This general minus sign is odd
    second_half = jnp.sum(spectrum_missing*phase)
    return jnp.reshape((first_half + second_half).real, (1,))

def fft_interp(
    k: jnp.ndarray, 
    spectrum: jnp.ndarray, 
    x: jnp.ndarray
) -> float:
    r"""Calculates the value of a field $`f`$ at an arbitrary position
    $`x`$ using 

    ```math
    f(x) = \sum_i \hat f e^{ik^\top x}
    ```

    for complex signals. The `spectrum` and the frequency coordinate grid $`k`$ must conform to 
    [`jax.numpy.fft.fftn`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.fftn.html).


    Args:
        k (jnp.ndarray): [description]
        spectrum (jnp.ndarray): [description]
        x (jnp.ndarray): [description]

    Returns:
        float: [description]
    """    
    return jnp.reshape(jnp.sum((spectrum)*jnp.exp(1j*jnp.sum(k*x,axis=-1))), (1,))

def make_filter_fun(domain: Domain, axis: Union[int, Tuple[int]]):
    r'''If axis is an integer, filtering is applied along the given axis. If
    instead axis is None, standard Fourier filtering is used. The size of the
    kernel should be 1D or ND accordingly.
    '''

    # if single axis, move it to the back and use broadcasting
    if isinstance(axis, int): 
        def filter_fun(x: jnp.ndarray, kernel: jnp.ndarray):
            if "complex" in str(x.dtype):
                ffts = [jnp.fft.fft, jnp.fft.ifft]
            else:
                ffts = [jnp.fft.rfft, jnp.fft.irfft]
            x = jnp.moveaxis(x, axis, -1)

            # Use real or complex fft
            Fx = ffts[0](x, axis=-1)
            dx = ffts[1](kernel * Fx, axis=-1, n=x.shape[-1])

            # Back to original axis
            dx = jnp.moveaxis(dx, -1, axis)
            return dx
    elif axis is None:
        # Standard FFT filtering
        domain_axes = list(range(-1, -domain.ndim, -1))

        def filter_fun(x: jnp.ndarray, kernel: jnp.ndarray):
            # Choose FFT and filter according to signal domain (real or complex)
            if "complex" in str(x.dtype):
                return jnp.fft.ifftn(
                    kernel * jnp.fft.fftn(x, axes=domain_axes), axes=domain_axes
                )
            else:
                return jnp.fft.irfftn(
                    kernel * jnp.fft.rfftn(x, axes=domain_axes), axes=domain_axes
                )
    else:
        raise ValueError("axis should be an integer or None, got {axis}")
    return filter_fun