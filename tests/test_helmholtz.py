from jax import numpy as jnp

from jwave.acoustics import ongrid_helmholtz_solver
from jwave.geometry import Domain, Medium


def test_if_homog_helmholtz_runs():
    N = (128,128)
    domain = Domain(N,(1.,1.))
    src_field = jnp.zeros(N).astype(jnp.complex64)
    src_field = src_field.at[64, 22].set(1.0)
    medium = Medium(
        domain,
        sound_speed=jnp.ones(N),
        density=jnp.ones(N),
        attenuation=None,
        pml_size=15
    )

    params, solver = ongrid_helmholtz_solver(
        medium,
        omega=1.,
        tol=1e-5,
        restart=5,
        method="gmres",
        maxiter=10,
        source = src_field
    )

    # Run simulation
    field = solver(params)

if __name__ == "__main__":
    test_if_homog_helmholtz_runs()
