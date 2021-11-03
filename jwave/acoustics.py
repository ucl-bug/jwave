from jwave import geometry
from jaxdf import ode
import jaxdf.operators as jops
from jaxdf.core import Field, operator
from jaxdf.discretization import (
    Coordinate,
    FourierSeries,
    StaggeredRealFourier,
    UniformField,
)
from jaxdf.utils import join_dicts
from typing import Callable, Union, Tuple

from jax import numpy as jnp
from jax.tree_util import tree_map
from jax.scipy.sparse.linalg import gmres, bicgstab


def _base_pml(
    transform_fun: Callable, medium: geometry.Medium, exponent=2.0, alpha_max=2.0
) -> jnp.ndarray:
    delta_pml = list(map(lambda x: x / 2 - medium.pml_size, medium.domain.size))
    delta_pml, delta_pml_f = UniformField(medium.domain, len(delta_pml)).from_scalar(
        jnp.asarray(delta_pml), "delta_pml"
    )
    coordinate_discr = Coordinate(medium.domain)
    X = Field(coordinate_discr, params={}, name="X")

    @operator()
    def X_pml(X, delta_pml):
        diff = (jops.elementwise(jnp.abs)(X) + (-1.0) * delta_pml) / (medium.pml_size)
        on_pml = jops.elementwise(lambda x: jnp.where(x > 0, x, 0))(diff)
        alpha = alpha_max * (on_pml ** exponent)
        exp_val = transform_fun(alpha)
        return exp_val

    outfield = X_pml(X=X, delta_pml=delta_pml_f)
    global_params = outfield.get_global_params()
    pml_val = outfield.get_field_on_grid(0)(
        global_params, {"X": {}, "delta_pml": delta_pml}
    )
    return pml_val


def complex_pml_on_grid(
    medium: geometry.Medium, omega: float, exponent=2.0, alpha_max=2.0
) -> jnp.ndarray:
    transform_fun = lambda alpha: 1.0 / (1 + 1j * alpha)
    return _base_pml(transform_fun, medium, exponent, alpha_max)


def td_pml_on_grid(
    medium: geometry.Medium, dt: float, exponent=2.0, alpha_max=2.0
) -> jnp.ndarray:
    transform_fun = lambda alpha: jops.elementwise(jnp.exp)((-1) * alpha * dt / 2)
    return _base_pml(transform_fun, medium, exponent, alpha_max)


def ongrid_wave_propagation(
    medium: geometry.Medium,
    time_array: geometry.TimeAxis,
    sources: geometry.Sources,
    discretization: str = "StaggeredFourier",
    sensors=None,
    output_t_axis=None,
    backprop=False,
    checkpoint=False,
) -> Tuple[dict, Callable]:
    r"""Constructs a wave propagation operator on a grid.

    The operator solves the equations

    ```math
    \begin{align}
        \frac{\partial u}{\partial t} &= - \frac{1}{\rho_0}\nabla p \\
        \frac{\partial \rho}{\partial t} &= -\rho_0\nabla \cdot u - u \cdot \nabla \rho_0 + S_M  \\
        p &= c_0^2(\rho + d\cdot\nabla \rho_0)
    \end{align}
    ```

    The function returns a tuple, where the first element is a 
    dictionary of parameters, and the second element is a function
    that takes a dictionary of parameters and returns the solution.

    The solution is defined for all time points in the time axis.

    If the `sensors` argument is not None, the function returns the
    entire field at each time point. Otherwise, the function returns
    the field at the sensors locations at each time point.

    Setting `backprop` to True will return a function that can be
    differentiated using backpropagation. Otherwise, only forward
    differentiation is supported.

    Lastly. if `checkpoint` is True, memory requirements are reduced
    for backpropagation, at the expense of a longer runtime.

    Args:
        medium (geometry.Medium): The acoustic medium
        time_array (geometry.TimeAxis): The time axis
        sources (geometry.Sources): Point sources
        discretization (str, optional): Numerical discretization method. Supported 
            discretizations are `'StaggeredFourier'` and `'Fourier'`. Defaults to "StaggeredFourier".
        sensors ([type], optional): [description]. Defaults to None.
        output_t_axis ([type], optional): [description]. Defaults to None.
        backprop (bool, optional): [description]. Defaults to False.
        checkpoint (bool, optional): [description]. Defaults to False.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        Tuple[dict, Callable]: [description]

    The structure of the dictionary of parameters returned is
    ```json
    {
        "shared": Internal operator parameters,
        "idependent": Internal operator parameters,
        "integrator": {
            "dt": timestep,
            "pml_grid": PML function on the grid,
        },
        "source_signals": Arrays of source signals,
        "acoustic_params": {
            "speed_of_sound": Speed of sound map,
            "density": Acoustic density map,
        },
        "initial_fields": {
            "rho": Initial acoustic density,
            "u": Initial acoustic velocity,
        },
    }
    ```
    The simulation function can be differentiated with respect to any of such
    parameters.

    !!! todo
        The structure of the internal parameters is a bit obscure at the moment,
        and will be improved in the future.
    """
    

    # TODO: This could be more flexible and accept custom discretizations

    if discretization == "StaggeredFourier":
        discretization = StaggeredRealFourier
    elif discretization == "Fourier":
        discretization = FourierSeries
    else:
        raise ValueError(
            f"Discretization {discretization} not supported. "
            "Supported discretizations are 'StaggeredFourier' and 'Fourier'."
        )

    # Setup parameters
    c_ref = jnp.amin(medium.sound_speed)
    dt = time_array.dt

    # Get steps to be saved
    if output_t_axis is None:
        output_t_axis = time_array
        t = jnp.arange(0, output_t_axis.t_end + output_t_axis.dt, output_t_axis.dt)
    else:
        t = jnp.arange(0, output_t_axis.t_end + output_t_axis.dt, output_t_axis.dt)
    output_steps = (t / dt).astype(jnp.int32)

    # Making PML on grid
    pml_grid = td_pml_on_grid(medium, dt)

    # Making math operators for ODE solver
    fwd_grad = jops.staggered_grad(c_ref, dt, geometry.Staggered.FORWARD)
    bwd_diag_jac = jops.staggered_diag_jacobian(c_ref, dt, geometry.Staggered.BACKWARD)

    @operator()
    def du(rho0, p):
        dp = fwd_grad(p)
        return (-1.0) * dp / rho0

    @operator()
    def drho(u, rho0, Source_m):
        du = bwd_diag_jac(u)
        return (-1.0) * rho0 * du + Source_m

    @operator()
    def p_new(c, rho):
        return (c ** 2.0) * jops.sum_over_dims(rho)

    # Defining field families
    discr_1D = discretization(medium.domain)
    discr_ND = discretization(medium.domain, dims=medium.domain.ndim)

    c, c_f = discr_1D.empty_field(name="c")
    p, p_f = discr_1D.empty_field(name="p")
    rho0, rho0_f = discr_1D.empty_field(name="rho")
    rho, rho_f = discr_ND.empty_field(name="rho0")
    SM, SM_f = discr_ND.empty_field(name="Source_m")
    u, u_f = discr_ND.empty_field(name="u")

    # Numerical functions
    # Note that we keep the shared dictionaries separate, to reduce
    # memory usage.
    # They need to be added when using functions
    _du = du(rho0=rho0_f, p=p_f)
    gp_du = _du.get_global_params()
    shared_params = gp_du["shared"].copy()
    del gp_du["shared"]

    def du_f(gp, rho0, p):
        return _du.get_field_on_grid(0)(gp, {"rho0": rho0, "p": p})

    _drho = drho(u=u_f, rho0=rho0_f, Source_m=SM_f)
    gp_drho = _drho.get_global_params()
    shared_params = join_dicts(shared_params, gp_drho["shared"].copy())
    del gp_drho["shared"]

    def drho_f(gp, u, rho0, Sm):
        return _drho.get_field_on_grid(0)(gp, {"u": u, "rho0": rho0, "Source_m": Sm})

    _p_new = p_new(c=c_f, rho=rho_f)
    gp_pnew = _p_new.get_global_params()
    shared_params = join_dicts(shared_params, gp_pnew["shared"].copy())
    del gp_pnew["shared"]

    def p_new_f(gp, c, rho):
        return _p_new.get_field_on_grid(0)(gp, {"c": c, "rho": rho})

    # Represents sensors as a measurement operator on the whole field
    measurement_operator = sensor_to_operator(sensors)

    # Defining source scaling function
    def src_to_field(source_signals, t):
        src = jnp.zeros(medium.domain.N)
        idx = (t / dt).round().astype(jnp.int32)
        signals = source_signals[:, idx] / len(medium.domain.N)
        src = src.at[sources.positions].add(signals)
        return jnp.expand_dims(src, -1)

    # Defining parameters
    params = {
        "shared": shared_params,
        "idependent": {
            "du_dt": gp_du,
            "drho_dt": gp_drho,
            "p_new": gp_pnew,
        },
        "integrator": {
            "dt": dt,
            "pml_grid": pml_grid,
        },
        "source_signals": sources.signals,
        "acoustic_params": {
            "speed_of_sound": jnp.expand_dims(medium.sound_speed, -1),
            "density": jnp.expand_dims(medium.density, -1),
        },
        "initial_fields": {
            "rho": rho_f.params,
            "u": u_f.params,
        },
    }

    # Semi-implicit solver functions
    def du_dt(params, rho, t):
        c = params["acoustic_params"]["speed_of_sound"]
        rho_0 = params["acoustic_params"]["density"]

        # Making pressure field
        gp = params["idependent"]["p_new"]
        gp["shared"] = params["shared"]
        p = p_new_f(gp, c, rho)

        # Making density update
        gp = params["idependent"]["du_dt"]
        gp["shared"] = params["shared"]
        output = du_f(gp, rho_0, p)
        return output

    def drho_dt(params, u, t):
        rho_0 = params["acoustic_params"]["density"]
        src = src_to_field(params["source_signals"], t)

        gp = params["idependent"]["drho_dt"]
        gp["shared"] = params["shared"]

        output = drho_f(gp, u, rho_0, src)
        return output

    # Defining solver
    def solver(params):
        return ode.generalized_semi_implicit_euler(
            params,
            du_dt,
            drho_dt,
            measurement_operator,
            params["integrator"]["pml_grid"],
            params["initial_fields"]["u"],
            params["initial_fields"]["rho"],
            params["integrator"]["dt"],
            output_steps,
            backprop,
            checkpoint,
        )

    return params, solver

def helmholtz_on_grid(
    medium: geometry.Medium,
    omega: float,
    source=None,
    discretization=FourierSeries,
):
    fourier_discr = discretization(medium.domain)

    # Initializing PML
    pml_grid = complex_pml_on_grid(medium, omega)

    # Modified laplacian
    rho0_val = medium.density
    if rho0_val is None:
        rho0_val = 1.
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
            0.0, name="alpha")
        k_fun = lambda u, omega, c, alpha: ((omega / c) ** 2) * u
    else:
        if (type(medium.attenuation) is float) or (type(medium.attenuation) is int):
            alpha_params, alpha = UniformField(medium.domain, dims=1).from_scalar(
            medium.attenuation, name="alpha")
        else:
            alpha_params, alpha = fourier_discr.from_array(
                medium.attenuation, 
                name="alpha",
                expand_params=True)

        def k_fun(u, omega, c, alpha):
            k_mod = (omega / c) ** 2 + 2j*(omega**3)*alpha/c
            return u*k_mod

    # Helmholtz operator
    @operator()
    def helmholtz(u, c, rho0, alpha, pml):
        # Get the modified laplacian
        L = laplacian(u, rho0, pml)

        # Add the wavenumber term
        k = k_fun(u, omega, c, alpha)
        return L + k

    # Discretizing operator
    u_fourier_params, u = fourier_discr.empty_field(name="u")
    
    if source is None:
        src_fourier_params, _ = fourier_discr.empty_field(name="src")
    else:
        src_fourier_params, _ = fourier_discr.from_array(
                source, 
                name="src",
                expand_params=True)

    if (type(rho0_val) is float) or (type(rho0_val) is int):
        rho0_params, rho0 = UniformField(medium.domain, dims=1).from_scalar(
            rho0_val, name="rho0")
    else:
        _, rho0 = fourier_discr.empty_field(name="rho0")
        rho0_params, rho0 = fourier_discr.from_array(
            medium.density, # +0j,
            name="rho0",
            expand_params=True)

    c_fourier_params, c = fourier_discr.empty_field(name="c")
    c_fourier_params = jnp.expand_dims(medium.sound_speed, -1)
    pml = Field(fourier_discr, params=pml_grid, name="pml")

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
        "source": src_fourier_params
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
) -> Tuple[dict, Callable]:
    
    params, f = helmholtz_on_grid(
        medium,
        omega,
        source=source,
        discretization=discretization,
    )
    params["solver_params"] = {"tol": tol, "maxiter": maxiter}

    def solver(params):
        def helm_func(u):
            return f(
                params["globals"], {
                    "u": u, 
                    "c": params["c"],
                    "rho0": params["rho0"], 
                    "alpha": params["alpha"],
                    "pml": params["pml"]
                }
            )

        if method == "gmres":
            return gmres(
                helm_func,
                params["source"],
                x0=params["guess"],
                tol=params["solver_params"]["tol"],
                restart=restart,
                maxiter=params["solver_params"]["maxiter"],
                solve_method=solve_method,
            )[0]
        elif method == "bicgstab":
            return bicgstab(
                helm_func,
                params["source"],
                x0=params["guess"],
                tol=params["solver_params"]["tol"],
                maxiter=params["solver_params"]["maxiter"],
            )[0]

    return params, solver


def sensor_to_operator(sensors):
    if sensors is None:

        def measurement_operator(x):
            return x  # identity operator

    elif isinstance(sensors, geometry.Sensors):
        # Define the application of the porjection matrix at the sensors
        # locations as a function
        if len(sensors.positions) == 1:

            def measurement_operator(x):
                return tree_map(lambda leaf: leaf[..., sensors.positions[0]], x)

        elif len(sensors.positions) == 2:

            def measurement_operator(x):
                return tree_map(
                    lambda leaf: leaf[..., sensors.positions[0], sensors.positions[1]],
                    x,
                )

        elif len(sensors.positions) == 3:

            def measurement_operator(x):
                return tree_map(
                    lambda leaf: leaf[
                        ...,
                        sensors.positions[0],
                        sensors.positions[1],
                        sensors.positions[2],
                    ],
                    x,
                )

        else:
            raise ValueError(
                "Sensors positions must be 1, 2 or 3 dimensional. Not {}".format(
                    len(sensors.positions)
                )
            )
    else:
        measurement_operator = sensors
    return measurement_operator
