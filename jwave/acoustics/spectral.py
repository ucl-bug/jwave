from jax import numpy as jnp
from jax.numpy import ndarray
from jaxdf.geometry import Domain


def kspace_op(
    domain: Domain,
    c_ref: float,
    dt: float,
) -> ndarray:
    r"""Returns the k-space operator for the given domain and reference
    speed of sound. The operator is defined as

    $$
    \kappa = \text{sinc}(c_{ref}k\Delta t/2)
    $$

    where $k$ is the wavenumber and $\Delta t$ is the time step.

    Args:
      domain (Domain): The domain to get the k-space operator for.
      c_ref (float): The reference speed of sound.
      dt (float): The time step.

    Returns:
      jnp.ndarray: The k-space operator.
    """

    # Get the frequency axis manually, since we
    # are nor using the rFFT
    # TODO: Implement operators with rFFT
    def f(N, dx):
        return jnp.fft.fftfreq(N, dx) * 2 * jnp.pi

    k_vec = [f(n, delta) for n, delta in zip(domain.N, domain.dx)]

    # Building k-space operator
    K = jnp.stack(jnp.meshgrid(*k_vec, indexing="ij"))
    k_magnitude = jnp.sqrt(jnp.sum(K**2, 0))
    k_space_op = jnp.sinc(c_ref * k_magnitude * dt / (2 * jnp.pi))
    parameters = {"k_vec": k_vec, "k_space_op": k_space_op}
    return parameters
