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
from jaxdf.discretization import FourierSeries, StaggeredRealFourier
from jaxdf.utils import join_dicts

from jwave import geometry
from jwave.acoustics.conversion import pressure_from_density
from jwave.acoustics.pml import td_pml_on_grid
from jwave.signal_processing import smooth
from jwave.utils import is_numeric

# Custom typedef
PyTree = Any


def ongrid_simple_wave_propagation(
    medium: geometry.Medium,
    time_array: geometry.TimeAxis,
    sources: geometry.Sources = None,
    discretization=StaggeredRealFourier,
    sensors=None,
    output_t_axis=None,
    backprop=False,
    checkpoint=False,
    u0=None,
    p0=None,
    u_update: Union[None, Callable] = None,
    p_update: Union[None, Callable] = None,
    rho_update: Union[None, Callable] = None,
) -> Tuple[dict, Callable]:
    r"""Constructs a wave propagation operator on a grid, for homogeneous
    density and no attenuation.

    The operator solves the equations

    ```math
    \begin{align}
        \frac{\partial u_i}{\partial t} &= (\nabla p)_i \\
        \frac{\partial \rho_i}{\partial t} &= -(\nabla \cdot u + S_M)_i \\
        \frac{\partial p}{\partial t} &= c_0^2 \sum_i \rho_i
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
        sources (geometry.Sources): Point sources. Defaults to `None`
        discretization (str, optional): Numerical discretization method. Supported 
            discretizations are `'StaggeredFourier'` and `'Fourier'`. Defaults to "StaggeredFourier".
        sensors ([type], optional): [description]. Defaults to None.
        output_t_axis ([type], optional): [description]. Defaults to None.
        backprop (bool, optional): [description]. Defaults to False.
        checkpoint (bool, optional): [description]. Defaults to False.
        u0 ([type], optional): Initial velocity field. Defaults to None.
        p0 ([type], optional): Initial pressure field. Defaults to None.
        u_update ([type], optional): If not None, overrides
            eq. (1). It must be a `jaxdf` operator accepting a field ($`p`$) 
            and returning a field. Defaults to None.
        p_update ([type], optional): If not None, overrides eq. (2). It must be a
            `jaxdf` operator accepting two fields ($`u, S_M`$) and returning a field. 
            Defaults to None.
        rho_update ([type], optional): If not None, overrides eq. (3). It must be a
            `jaxdf` operator accepting two fields ($`\rho, c`$) and returning a field.


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

    # Setup parameters
    c_ref = jnp.amax(medium.sound_speed)
    dt = time_array.dt

    # Get steps to be saved
    if output_t_axis is None:
        output_t_axis = time_array
        t = jnp.arange(0, output_t_axis.t_end + output_t_axis.dt, output_t_axis.dt)
    else:
        t = jnp.arange(0, output_t_axis.t_end + output_t_axis.dt, output_t_axis.dt)
    output_steps = (t / dt).astype(jnp.int32)

    # Making PML on grid
    pml_grid = td_pml_on_grid(
        medium, dt, c0=c_ref, dx=medium.domain.dx[0]
    )

    # Making math operators for ODE solver
    fwd_grad = jops.staggered_grad(c_ref, dt, geodf.Staggered.FORWARD)
    bwd_diag_jac = jops.staggered_diag_jacobian(c_ref, dt, geodf.Staggered.BACKWARD)

    # Define forward model
    if u_update is None:

        @operator()
        def du(p):
            dp = fwd_grad(p)
            return (-1.0) * dp

    else:
        du = operator()(u_update)

    if rho_update is None:

        @operator()
        def drho(u, Source_m):
            du = bwd_diag_jac(u)
            return (-1.0) * du + Source_m

    else:
        drho = operator()(rho_update)

    if p_update is None:

        @operator()
        def p_new(c, rho):
            return (c ** 2.0) * jops.sum_over_dims(rho)

    else:
        p_new = operator()(p_update)

    # Defining field families
    discr_1D = discretization(medium.domain)
    discr_ND = discretization(medium.domain, dims=medium.domain.ndim)

    _, c_f = discr_1D.empty_field(name="c")
    _, p_f = discr_1D.empty_field(name="p")
    _, rho_f = discr_ND.empty_field(name="rho")
    _, SM_f = discr_ND.empty_field(name="Source_m")
    _, u_f = discr_ND.empty_field(name="u")

    # Update initial fields
    if u0 is not None:
        u_f.params = u0
    if p0 is None:
        p0 = jnp.zeros_like(medium.sound_speed)

    # Numerical functions
    # Note that we keep the shared dictionaries separate, to reduce
    # memory usage.
    # They need to be added when using functions
    _du = du(p=p_f)
    gp_du = _du.get_global_params()
    shared_params = gp_du["shared"].copy()
    del gp_du["shared"]

    def du_f(gp, p):
        return _du.get_field_on_grid(0)(gp, {"p": p})

    _drho = drho(u=u_f, Source_m=SM_f)
    gp_drho = _drho.get_global_params()
    shared_params = join_dicts(shared_params, gp_drho["shared"].copy())
    del gp_drho["shared"]

    def drho_f(gp, u, Sm):
        return _drho.get_field_on_grid(0)(gp, {"u": u, "Source_m": Sm})

    _p_new = p_new(c=c_f, rho=rho_f)
    gp_pnew = _p_new.get_global_params()
    shared_params = join_dicts(shared_params, gp_pnew["shared"].copy())
    del gp_pnew["shared"]

    def p_new_f(gp, c, rho):
        return _p_new.get_field_on_grid(0)(gp, {"c": c, "rho": rho})

    # Represents sensors as a measurement operator on the whole field
    measurement_operator = sensor_to_operator(sensors)

    # Defining source scaling function
    if sources is not None:
        sources = geometry.Sources.no_sources(medium.domain)

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
        "source_signals": sources.signals if sources is geometry.Sources else [],
        "acoustic_params": {
            "speed_of_sound": jnp.expand_dims(medium.sound_speed, -1),
            "density": jnp.expand_dims(medium.density, -1),
        },
        "initial_fields": {
            "p": p0,
            "u": u_f.params,
        },
    }

    # Semi-implicit solver functions
    def du_dt(params, rho, t):
        c = params["acoustic_params"]["speed_of_sound"]

        # Making pressure field
        gp = params["idependent"]["p_new"]
        gp["shared"] = params["shared"]
        p = p_new_f(gp, c, rho)

        # Making density update
        gp = params["idependent"]["du_dt"]
        gp["shared"] = params["shared"]
        output = du_f(gp, p)
        return output

    def drho_dt(params, u, t):
        if sources is not None:
            src = sources.to_field(params["source_signals"], t)
        else:
            src = 0.0

        gp = params["idependent"]["drho_dt"]
        gp["shared"] = params["shared"]

        output = drho_f(gp, u, src)
        return output

    # Defining solver
    def solver(params):
        # Make initial density from pressure field
        p0 = params["initial_fields"]["p"]
        p0 = smooth(p0)
        rho0 = p0 / (params["acoustic_params"]["speed_of_sound"][..., 0] ** 2.0)
        rho0 = jnp.stack([rho0] * medium.domain.ndim, -1) / medium.domain.ndim

        # Integrate
        sensors_data = ode.generalized_semi_implicit_euler(
            params,
            du_dt,
            drho_dt,
            measurement_operator,
            params["integrator"]["pml_grid"],
            params["initial_fields"]["u"],
            rho0,
            params["integrator"]["dt"],
            output_steps,
            backprop,
            checkpoint,
        )

        # Get pressure from density
        p = pressure_from_density(
            sensors_data, params["acoustic_params"]["speed_of_sound"][..., 0], sensors
        )
        return p

    return params, solver


def ongrid_wave_propagation(
    medium: geometry.Medium,
    time_array: geometry.TimeAxis,
    sources: geometry.Sources = None,
    discretization=StaggeredRealFourier,
    sensors=None,
    output_t_axis=None,
    backprop=False,
    checkpoint=False,
    u0=None,
    p0=None,
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
        sources (geometry.Sources): Point sources. Defaults to `None`
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

    # Setup parameters
    c_ref = jnp.amax(medium.sound_speed)
    dt = time_array.dt

    # Get steps to be saved
    if output_t_axis is None:
        output_t_axis = time_array
        t = jnp.arange(0, output_t_axis.t_end + output_t_axis.dt, output_t_axis.dt)
    else:
        t = jnp.arange(0, output_t_axis.t_end + output_t_axis.dt, output_t_axis.dt)
    output_steps = (t / dt).astype(jnp.int32)

    # Making PML on grid
    pml_grid = td_pml_on_grid(
        medium, dt, c0=c_ref, dx=medium.domain.dx[0]
    )

    # Making math operators for ODE solver
    fwd_grad = jops.staggered_grad(c_ref, dt, geodf.Staggered.FORWARD)
    bwd_diag_jac = jops.staggered_diag_jacobian(c_ref, dt, geodf.Staggered.BACKWARD)

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

    _, c_f = discr_1D.empty_field(name="c")
    _, p_f = discr_1D.empty_field(name="p")
    _, rho0_f = discr_1D.empty_field(name="rho")
    _, rho_f = discr_ND.empty_field(name="rho0")
    _, SM_f = discr_ND.empty_field(name="Source_m")
    _, u_f = discr_ND.empty_field(name="u")

    # Update initial fields
    if u0 is not None:
        u_f.params = u0
    if p0 is None:
        p0 = jnp.zeros_like(medium.sound_speed)

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
    if sources is not None:

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
        "source_signals": sources.signals if sources is not None else [],
        "acoustic_params": {
            "speed_of_sound": jnp.expand_dims(medium.sound_speed, -1),
            "density": jnp.expand_dims(medium.density, -1),
        },
        "initial_fields": {
            "p": p0,
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
        if sources is not None:
            src = src_to_field(params["source_signals"], t)
        else:
            src = 0.0

        gp = params["idependent"]["drho_dt"]
        gp["shared"] = params["shared"]

        output = drho_f(gp, u, rho_0, src)
        return output

    # Defining solver
    def solver(params):
        # Make initial density from pressure field
        p0 = params["initial_fields"]["p"]
        p0 = smooth(p0)
        rho0 = p0 / (params["acoustic_params"]["speed_of_sound"][..., 0] ** 2.0)
        rho0 = jnp.stack([rho0] * medium.domain.ndim, -1) / medium.domain.ndim

        # Integrate
        sensors_data = ode.generalized_semi_implicit_euler(
            params,
            du_dt,
            drho_dt,
            measurement_operator,
            params["integrator"]["pml_grid"],
            params["initial_fields"]["u"],
            rho0,
            params["integrator"]["dt"],
            output_steps,
            backprop,
            checkpoint,
        )

        # Get pressure from density
        p = pressure_from_density(
            sensors_data, params["acoustic_params"]["speed_of_sound"][..., 0], sensors
        )
        return p

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
                return tree_map(lambda leaf: leaf[sensors.positions[0]], x)

        elif len(sensors.positions) == 2:

            def measurement_operator(x):
                return tree_map(
                    lambda leaf: leaf[sensors.positions[0], sensors.positions[1]],
                    x,
                )

        elif len(sensors.positions) == 3:

            def measurement_operator(x):
                return tree_map(
                    lambda leaf: leaf[
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
