from jax import numpy as jnp

from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave.geometry import Domain, FourierSeries, Medium


def test_if_homog_helmholtz_runs():
    N = (128,128)
    domain = Domain(N,(1.,1.))
    src_field = jnp.zeros(N).astype(jnp.complex64)
    src_field = src_field.at[64, 22].set(1.0)
    src_field = jnp.expand_dims(src_field, axis=-1)
    src_field = FourierSeries(src_field, domain)

    medium = Medium(
        domain,
        sound_speed=1.0,
        pml_size=15
    )

    field = helmholtz_solver(
        medium,
        1.,
        src_field,
        tol=1e-5,
        restart=5,
        method="gmres",
        maxiter=10,
    )

if __name__ == "__main__":
    test_if_homog_helmholtz_runs()
