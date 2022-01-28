from dataclasses import dataclass
from typing import Any, Callable, Tuple, Union

import jax
import jaxdf.operators as jops
from jax import numpy as jnp
from jax.scipy.sparse.linalg import bicgstab, gmres
from jax.tree_util import tree_map
from jaxdf import geometry as geodf
from jaxdf import ode
from jaxdf.core import Field, operator
from jaxdf.discretization import (Coordinate, FourierSeries,
                                  StaggeredRealFourier, UniformField)
from jaxdf.utils import join_dicts

from jwave import geometry
from jwave.acoustics.conversion import pressure_from_density
from jwave.acoustics.pml import complex_pml_on_grid
from jwave.acoustics.time_varying import sensor_to_operator
from jwave.signal_processing import smooth
from jwave.utils import is_numeric

# Custom typedef
PyTree = Any


def helmholtz_on_grid(
    medium: geometry.Medium,
    omega: float,
    source: Union[None, jnp.ndarray] = None,
    discretization=FourierSeries,
) -> Union[PyTree, Callable]:
    """
    Constructs the Helmholtz operator on a homogeneous collocation
    grid. The operator is returned as couple of parameters and
    callable

    Args:
        medium (geometry.Medium): Medium object
        omega (float): Angular frequency
        source (jnp.ndarray): Source map (optional)
        discretization (Discretization): Discretization object

    Returns:
        params (dict): Parameters of the Helmholtz operator
        helmholtz (Callable): Helmholtz operator

    The discretization object must be compatible with the following operations,
    on top of the standard arithmetic ones:
        - `jops.gradient()`
        - `jops.diag_jacobian()`
        - `jops.sum_over_dims()`

    !!! example
        ```python
        from jwave import geometry
        from jwave.acoustics import helmholtz_on_grid

        domain = Domain((128, 256), (1., 1.))
        medium = geometry.Medium(
            domain,
            speed_of_sound=jnp.ones(128,128)
        )

        params, solver = helmholtz_on_grid(medium, omega=1.)
        ```
    """
    discretization = discretization(medium.domain)

    # Initializing PML
    pml_grid = complex_pml_on_grid(medium, omega)

    # Modified laplacian
    rho0_val = medium.density
    if rho0_val is None:
        rho0_val = 1.0
    if (type(rho0_val) is float) or (type(rho0_val) is int):

        def laplacian(u, rho0, pml):
            grad_u = jops.gradient(u)
            mod_grad_u = grad_u * pml
            mod_diag_jacobian = jops.diag_jacobian(mod_grad_u) * pml
            return jops.sum_over_dims(mod_diag_jacobian)

    else:
        assert rho0_val.ndim == medium.sound_speed.ndim

        def laplacian(u, rho0, pml):
            grad_u = jops.gradient(u)
            mod_grad_u = grad_u * pml
            mod_diag_jacobian = jops.diag_jacobian(mod_grad_u) * pml
            nabla_u = jops.sum_over_dims(mod_diag_jacobian)

            grad_rho0 = jops.gradient(rho0)
            rho_u = jops.sum_over_dims(mod_grad_u * grad_rho0) / rho0

            return nabla_u - rho_u

    # Absorption term
    if medium.attenuation is None:
        alpha_params, alpha = UniformField(medium.domain, dims=1).from_scalar(
            0.0, name="alpha"
        )
        k_fun = lambda u, omega, c, alpha: ((omega / c) ** 2) * u
    else:
        if is_numeric(medium.attenuation):
            alpha_params, alpha = UniformField(medium.domain, dims=1).from_scalar(
                medium.attenuation, name="alpha"
            )
        else:
            alpha_params, alpha = discretization.from_array(
                medium.attenuation, name="alpha", expand_params=True
            )

        def k_fun(u, omega, c, alpha):
            k_mod = (omega / c) ** 2 + 2j * (omega ** 3) * alpha / c
            return u * k_mod

    # Helmholtz operator
    @operator()
    def helmholtz(u, c, rho0, alpha, pml):
        # Get the modified laplacian
        L = laplacian(u, rho0, pml)

        # Add the wavenumber term
        k = k_fun(u, omega, c, alpha)
        return L + k

    # Discretizing operator
    u_fourier_params, u = discretization.empty_field(name="u")

    if source is None:
        src_fourier_params, _ = discretization.empty_field(name="src")
    else:
        src_fourier_params, _ = discretization.from_array(
            source, name="src", expand_params=True
        )

    if is_numeric(rho0_val):
        rho0_params, rho0 = UniformField(medium.domain, dims=1).from_scalar(
            rho0_val, name="rho0"
        )
    else:
        _, rho0 = discretization.empty_field(name="rho0")
        rho0_params, rho0 = discretization.from_array(
            medium.density, name="rho0", expand_params=True  # +0j,
        )

    c_fourier_params, c = discretization.empty_field(name="c")
    c_fourier_params = jnp.expand_dims(medium.sound_speed, -1)
    pml = Field(discretization, params=pml_grid, name="pml")

    Hu = helmholtz(u=u, c=c, rho0=rho0, alpha=alpha, pml=pml)
    global_params = Hu.get_global_params()
    f = Hu.get_field_on_grid(0)

    # Parameters
    params = {
        "globals": global_params,
        "guess": u_fourier_params,
        "c": c_fourier_params,
        "rho0": rho0_params,
        "alpha": alpha_params,
        "pml": pml_grid,
        "source": src_fourier_params,
    }
    return params, f


def ongrid_helmholtz_solver(
    medium: geometry.Medium,
    omega: float,
    source=None,
    discretization=FourierSeries,
    method="gmres",
    restart=10,
    tol=1e-5,
    solve_method="batched",
    maxiter=None,
    checkpoint=True,
) -> Tuple[dict, Callable]:
    r"""
    Generates a solver for the Helmholtz equation on a grid.

    The equation solved is

    ```math
    -\frac{\omega^2}{c_0^2}P = \nabla^2 P - \frac{1}{\rho_0} \nabla \rho_0 \cdot \nabla P + \frac{2i\omega^3\alpha_0}{c_0} P - i \omega S_M.
    ```

    where `P` is the pressure, `\rho_0` is the background density,
    `c_0` is the background sound speed, `\alpha_0` is the attenuation,
    `\omega` is the angular frequency, and `S_M` is the mass source term.

    Args:
        medium (geometry.Medium): The acoustic medium.
        omega (float): The angular frequency.
        source (jnp.ndarray, optional): The source term.
        discretization (jaxdf.discretization, optional): The discretization to use. Defaults to FourierSeries.
        method (str, optional): The linear solver to use. Defaults to "gmres".
        restart (int, optional): The number of iterations to restart. Defaults to 10.
        tol (float, optional): The tolerance for the linear solver. Defaults to 1e-5.
        solve_method (str, optional): *Only for GMRES*. Defaults to "batched".
        maxiter (int, optional): Defaults to None (i.e. no maximum number of iterations).
        checkpoint (bool, optional): Wehther to checkpoint the forward solve for saving memory. Defaults to False.
    """

    params, f = helmholtz_on_grid(
        medium,
        omega,
        source=source,
        discretization=discretization,
    )
    params["solver_params"] = {"tol": tol, "maxiter": maxiter}

    def solver(params):
        @jax.jit
        def helm_func(u):
            return f(
                params["globals"],
                {
                    "u": u,
                    "c": params["c"],
                    "rho0": params["rho0"],
                    "alpha": params["alpha"],
                    "pml": params["pml"],
                },
            )

        if checkpoint:
            helm_func = jax.checkpoint(helm_func)

        residual_magnitude = jnp.linalg.norm(params["source"])
        iterations = 0
        x0 = params["guess"]

        # Scale the tolerance by the residual magnitude
        tol = params["solver_params"]["tol"]
        tol = tol * residual_magnitude

        while residual_magnitude > tol and iterations < params["solver_params"]["maxiter"]:
            if method == "gmres":
                x0 =  gmres(
                    helm_func,
                    params["source"],
                    x0=x0,
                    tol=0.0,
                    restart=restart,
                    maxiter=1,
                    solve_method=solve_method,
                )[0]
            elif method == "bicgstab":
                x0 = bicgstab(
                    helm_func,
                    params["source"],
                    x0=x0,
                    tol=0,
                    maxiter=restart,
                )[0]

            residual_magnitude = jnp.linalg.norm(helm_func(x0) - params["source"])
            iterations += 1

            # Print iteration info
            print(
                f"Iteration {iterations}: residual magnitude = {residual_magnitude}, tol = {tol:.2e}",
                flush=True,
            )
        
        return x0

    return params, solver

def ongrid_helmholtz_solver_verbose(
    medium: geometry.Medium,
    omega: float,
    source=None,
    discretization=FourierSeries,
    method="gmres",
    restart=10,
    tol=1e-5,
    solve_method="batched",
    maxiter=None,
) -> Tuple[dict, Callable]:
    r"""
    Generates a solver for the Helmholtz equation on a grid.

    The equation solved is

    ```math
    -\frac{\omega^2}{c_0^2}P = \nabla^2 P - \frac{1}{\rho_0} \nabla \rho_0 \cdot \nabla P + \frac{2i\omega^3\alpha_0}{c_0} P - i \omega S_M.
    ```

    where `P` is the pressure, `\rho_0` is the background density,
    `c_0` is the background sound speed, `\alpha_0` is the attenuation,
    `\omega` is the angular frequency, and `S_M` is the mass source term.

    Args:
        medium (geometry.Medium): The acoustic medium.
        omega (float): The angular frequency.
        source (jnp.ndarray, optional): The source term.
        discretization (jaxdf.discretization, optional): The discretization to use. Defaults to FourierSeries.
        method (str, optional): The linear solver to use. Defaults to "gmres".
        restart (int, optional): The number of iterations to restart. Defaults to 10.
        tol (float, optional): The tolerance for the linear solver. Defaults to 1e-5.
        solve_method (str, optional): *Only for GMRES*. Defaults to "batched".
        maxiter (int, optional): Defaults to None (i.e. no maximum number of iterations).
        checkpoint (bool, optional): Wehther to checkpoint the forward solve for saving memory. Defaults to False.
    """

    params, f = helmholtz_on_grid(
        medium,
        omega,
        source=source,
        discretization=discretization,
    )
    params["solver_params"] = {"tol": tol, "maxiter": maxiter}

    def solver(params):
        @jax.jit
        def helm_func(u):
            return f(
                params["globals"],
                {
                    "u": u,
                    "c": params["c"],
                    "rho0": params["rho0"],
                    "alpha": params["alpha"],
                    "pml": params["pml"],
                },
            )

        residual_magnitude = jnp.sqrt(jnp.sum(jnp.abs(params["source"])**2))
        iterations = 0
        x0 = params["guess"]

        # Scale the tolerance by the residual magnitude
        tol = params["solver_params"]["tol"]
        tol = tol * residual_magnitude

        while residual_magnitude > tol and iterations < params["solver_params"]["maxiter"]:
            if method == "gmres":
                x0 =  gmres(
                    helm_func,
                    params["source"],
                    x0=x0,
                    tol=0.0,
                    restart=restart,
                    maxiter=1,
                    solve_method=solve_method,
                )[0]
            elif method == "bicgstab":
                x0 = bicgstab(
                    helm_func,
                    params["source"],
                    x0=x0,
                    tol=0,
                    maxiter=restart,
                )[0]

            residual = helm_func(x0) - params["source"]
            residual_magnitude = jnp.sqrt(jnp.sum(jnp.abs(residual)**2))
            iterations += 1

            # Print iteration info
            print(
                f"Iteration {iterations}: residual magnitude = {residual_magnitude}, tol = {tol:.2e}",
                flush=True,
            )
        
        return x0

    return params, solver
