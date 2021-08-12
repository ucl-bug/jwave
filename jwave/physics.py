from jwave import geometry, ode
import jwave.operators as jops
from jwave.core import Field, operator
from jwave.discretization import Coordinate, StaggeredRealFourier, UniformField
from jwave.utils import join_dicts

from jax import numpy as jnp
from jax.tree_util import tree_map


def td_pml_on_grid(medium: geometry.Medium, dt: float, exponent=2., alpha_max=2.):
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
        exp_val = jops.elementwise(jnp.exp)((-1) * alpha * dt / 2)
        return exp_val

    outfield = X_pml(X=X, delta_pml=delta_pml_f)
    global_params = outfield.get_global_params()
    pml_val = outfield.get_field_on_grid(0)(
        global_params, {"X": {}, "delta_pml": delta_pml}
    )
    return pml_val


def ongrid_wave_propagation(
    medium: geometry.Medium,
    time_array: geometry.TimeAxis,
    sources: geometry.Sources,
    discretization= StaggeredRealFourier,
    sensors=None,
    output_t_axis=None,
    backprop=False,
    checkpoint=False,
):
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

        output =  drho_f(gp, u, rho_0, src)
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
