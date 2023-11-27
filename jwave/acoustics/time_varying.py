# This file is part of j-Wave.
#
# j-Wave is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# j-Wave is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with j-Wave. If not, see <https://www.gnu.org/licenses/>.

from typing import Callable, Dict, Tuple, TypeVar, Union

import equinox as eqx
import numpy as np
from jax import checkpoint as jax_checkpoint
from jax import numpy as jnp
from jax.lax import scan
from jaxdf import Field, operator
from jaxdf.discretization import FourierSeries, Linear, OnGrid
from jaxdf.mods import Module
from jaxdf.operators import diag_jacobian, shift_operator, sum_over_dims

from jwave.acoustics.spectral import kspace_op
from jwave.geometry import Medium, Sources, TimeAxis
from jwave.logger import logger
from jwave.signal_processing import smooth

from .pml import td_pml_on_grid

Any = TypeVar("Any")


class TimeWavePropagationSettings(Module):
    """
    TimeWavePropagationSettings configures the settings for
    time domain wave solvers. This class serves as a container
    for settings that influence how wave propagation is
    simulated.

    !!! example
    ```python
    >>> settings = TimeWavePropagationSettings(
    ...    c_ref = lambda m: m.min_sound_speed)
    >>> print(settings.checkpoint)
    True

    ```
    """

    c_ref: Callable = eqx.field(static=True)
    checkpoint: bool = eqx.field(static=True)
    smooth_initial: bool = eqx.field(static=True)

    def __init__(
        self,
        c_ref: Callable = lambda m: m.max_sound_speed,
        checkpoint: bool = True,
        smooth_initial: bool = True,
    ):
        """
        Initializes a new instance of the TimeWavePropagationSettings class.

        Args:
            c_ref (Callable, static): A callable that determines
                the reference speed of the wave solver. This is a
                expected to be a function that takes the `medium`
                variable and returns the reference sound speed
            checkpoint (bool, static): Flag indicating whether to
                use checkpointing to save memory during backpropagation.
                Defaults to True.
            smooth_initial (bool, static): Flag to determine
                whether to smooth initial pressure and velocity
                fields. Defaults to True.
        """
        self.c_ref = c_ref
        self.checkpoint = checkpoint
        self.smooth_initial = smooth_initial



def _shift_rho(rho0, direction, dx):
    if isinstance(rho0, OnGrid):
        rho0_params = rho0.params[..., 0]

        def linear_interp(u, axis):
            return 0.5 * (jnp.roll(u, -direction, axis) + u)

        rho0 = jnp.stack(
            [linear_interp(rho0_params, n) for n in range(rho0.domain.ndim)],
            axis=-1)
    elif isinstance(rho0, Field):
        rho0 = shift_operator(rho0, direction * dx)
    else:
        pass
    return rho0


@operator
def momentum_conservation_rhs(p: OnGrid,
                              u: OnGrid,
                              medium: Medium,
                              *,
                              c_ref=1.0,
                              dt=1.0,
                              params=None) -> OnGrid:
    r"""Staggered implementation of the momentum conservation equation.

    Args:
      p (OnGrid): The pressure field.
      u (OnGrid): The velocity field.
      medium (Medium): The medium.
      c_ref (float): The reference sound speed. **Unused**
      dt (float): The time step. **Unused**
      params: The operator parameters. **Unused**

    Returns:
      OnGrid: The right hand side of the momentum conservation equation.
    """
    # Staggered implementation
    dx = np.asarray(u.domain.dx)
    rho0 = _shift_rho(medium.density, 1, dx)
    dp = diag_jacobian(p, stagger=[0.5])
    return -dp / rho0


@operator
def momentum_conservation_rhs(
    p: FourierSeries,
    u: FourierSeries,
    medium: Medium,
    *,
    c_ref=1.0,
    dt=1.0,
    params=None,
) -> FourierSeries:
    r"""Staggered implementation of the momentum conservation equation.

    Args:
      p (FourierSeries): The pressure field.
      u (FourierSeries): The velocity field.
      medium (Medium): The medium.
      c_ref (float): The reference sound speed. Used to calculate the k-space operator.
      dt (float): The time step. Used to calculate the k-space operator.
      params: The operator parameters.

    Returns:
      FourierSeries: The right hand side of the momentum conservation equation.
    """
    if params == None:
        params = kspace_op(p.domain, c_ref, dt)

    dx = np.asarray(u.domain.dx)
    direction = 1

    # Shift rho
    rho0 = _shift_rho(medium.density, direction, dx)

    # Take a shifted gradient of the pressure
    k_vec = params["k_vec"]
    k_space_op = params["k_space_op"]

    shift_and_k_op = [
        1j * k * jnp.exp(1j * k * direction * delta / 2)
        for k, delta in zip(k_vec, dx)
    ]

    p_params = p.params[..., 0]
    Fu = jnp.fft.fftn(p_params)

    def single_grad(axis):
        Fx = jnp.moveaxis(Fu, axis, -1)
        k_op = jnp.moveaxis(k_space_op, axis, -1)
        iku = jnp.moveaxis(Fx * shift_and_k_op[axis] * k_op, -1, axis)
        return jnp.fft.ifftn(iku).real

    dp = jnp.stack([single_grad(i) for i in range(p.domain.ndim)], axis=-1)
    update = -p.replace_params(dp) / rho0

    return update


@operator
def mass_conservation_rhs(p: OnGrid,
                          u: OnGrid,
                          mass_source: object,
                          medium: Medium,
                          *,
                          c_ref,
                          dt,
                          params=None) -> OnGrid:
    r"""Implementation of the mass conservation equation. The pressure field
    is assumed to be staggered forward for each component, and will be staggered
    backward before being multiplied by the ambient density.

    Args:
      p (OnGrid): The pressure field.
      u (OnGrid): The velocity field.
      mass_source (object): The mass source.
      medium (Medium): The medium.
      c_ref (float): The reference sound speed. **Unused**
      dt (float): The time step. **Unused**
      params: The operator parameters. **Unused**

    Returns:
      OnGrid: The right hand side of the mass conservation equation.
    """

    rho0 = medium.density
    c0 = medium.sound_speed
    dx = np.asarray(p.domain.dx)

    # Staggered implementation
    du = diag_jacobian(u, stagger=[-0.5])
    update = -du * rho0 + 2 * mass_source / (c0 * p.domain.ndim * dx)
    return update, params


@operator
def mass_conservation_rhs(
    p: FourierSeries,
    u: FourierSeries,
    mass_source: object,
    medium: Medium,
    *,
    c_ref,
    dt,
    params=None,
) -> FourierSeries:
    r"""Implementation of the mass conservation equation. The pressure field
    is assumed to be staggered forward for each component, and will be staggered
    backward before being multiplied by the ambient density.

    Args:
      p (FourierSeries): The pressure field.
      u (FourierSeries): The velocity field.
      mass_source (object): The mass source.
      medium (Medium): The medium.
      c_ref (float): The reference sound speed. Used to calculate the k-space operator.
      dt (float): The time step. Used to calculate the k-space operator.
      params: The operator parameters.

    Returns:
      FourierSeries: The right hand side of the mass conservation equation.
    """

    if params == None:
        params = kspace_op(p.domain, c_ref, dt)

    dx = np.asarray(p.domain.dx)
    direction = -1

    k_vec = params["k_vec"]
    k_space_op = params["k_space_op"]
    rho0 = medium.density
    c0 = medium.sound_speed

    # Add shift to k vector
    shift_and_k_op = [
        1j * k * jnp.exp(1j * k * direction * delta / 2)
        for k, delta in zip(k_vec, dx)
    ]

    def single_grad(axis, u):
        Fu = jnp.fft.fftn(u)
        Fx = jnp.moveaxis(Fu, axis, -1)
        k_op = jnp.moveaxis(k_space_op, axis, -1)
        iku = jnp.moveaxis(Fx * shift_and_k_op[axis] * k_op, -1, axis)
        return jnp.fft.ifftn(iku).real

    du = jnp.stack(
        [single_grad(i, u.params[..., i]) for i in range(p.domain.ndim)],
        axis=-1)
    update = -p.replace_params(du) * rho0 + 2 * mass_source / (
        c0 * p.domain.ndim * dx)

    return update


@operator
def pressure_from_density(rho: Field, medium: Medium, *, params=None) -> Field:
    r"""Calculate pressure from acoustic density given by the raw output of the
    timestepping scheme.

    Args:
      rho (Field): The density field.
      medium (Medium): The medium.
      params: The operator parameters. **Unused**

    Returns:
      Field: The pressure field.
    """
    rho_sum = sum_over_dims(rho)
    c0 = medium.sound_speed
    return (c0**2) * rho_sum


@operator
def wave_propagation_symplectic_step(
    p: Linear,
    u: Linear,
    rho: Linear,
    medium: Medium,
    sources: Union[None, Sources],
    pmls: Dict,
    *,
    step: Union[int, object],
    c_ref=Union[None, object],
    dt=Union[None, object],
    params=None,
) -> Tuple[Linear, Linear, Linear]:

    # Evaluate mass source
    if sources is None:
        mass_src_field = 0.0
    else:
        mass_src_field = sources.on_grid(step)

    # Calculate momentum conservation equation
    du = momentum_conservation_rhs(p, u, medium, c_ref, dt)
    pml_u = pmls["pml_u"]
    u = pml_u * (pml_u * u + dt * du)

    # Calculate mass conservation equation
    drho = mass_conservation_rhs(p, u, mass_src_field, medium, c_ref, dt)
    pml_rho = pmls["pml_rho"]
    rho = pml_rho * (pml_rho * rho + dt * drho)

    # Update pressure
    p = pressure_from_density(rho, medium)

    # Return updated fields
    return [p, u, rho]


def ongrid_wave_prop_params(
    medium: OnGrid,
    time_axis: TimeAxis,
    *,
    settings: TimeWavePropagationSettings,
    **kwargs,
):
    # Check which elements of medium are a field
    x = [
        x for x in [medium.sound_speed, medium.density, medium.attenuation]
        if isinstance(x, Field)
    ][0]

    dt = time_axis.dt

    # Use settings to determine reference sound speed
    c_ref = settings.c_ref(medium)

    # Making PML on grid for rho and u
    def make_pml(staggering=0.0):
        pml_grid = td_pml_on_grid(medium,
                                  dt,
                                  c0=c_ref,
                                  dx=medium.domain.dx[0],
                                  coord_shift=staggering)
        pml = x.replace_params(pml_grid)
        return pml

    pml_rho = make_pml()
    pml_u = make_pml(staggering=0.5)

    return {
        "pml_rho": pml_rho,
        "pml_u": pml_u,
        "c_ref": c_ref,
    }


@operator(init_params=ongrid_wave_prop_params)
def simulate_wave_propagation(
    medium: Medium[OnGrid],
    time_axis: TimeAxis,
    *,
    settings: TimeWavePropagationSettings = TimeWavePropagationSettings(),
    sources=None,
    sensors=None,
    u0=None,
    p0=None,
    params=None,
):
    r"""Simulate the wave propagation operator.

    Args:
      medium (OnGridOrScalars): The medium.
      time_axis (TimeAxis): The time axis.
      sources (Any): The source terms. It can be any jax traceable object that
        implements the method `sources.on_grid(n)`, which returns the source
        field at the nth time step.
      sensors (Any): The sensor terms. It can be any jax traceable object that
        can be called as `sensors(p,u,rho)`, where `p` is the pressure field,
        `u` is the velocity field, and `rho` is the density field. The return
        value of this function is the recorded field. If `sensors` is not
        specified, the recorded field is the entire pressure field.
      u0 (Field): The initial velocity field. If `None`, the initial velocity
        field is set depending on the `p0` value, such that `u(t=0)=0`. Note that
        the velocity field is staggered forward by half time step relative to the
        pressure field.
      p0 (Field): The initial pressure field. If `None`, the initial pressure
        field is set to zero.
      checkpoint (bool): Whether to checkpoint the simulation at each time step.
        See [jax.checkpoint](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)
      smooth_initial (bool): Whether to smooth the initial conditions.
      params: The operator parameters.

    Returns:
      Any: The recording of the sensors at each time step.
    """

    # Default sensors simply return the presure field
    if sensors is None:
        sensors = lambda p, u, rho: p

    # Setup parameters
    output_steps = jnp.arange(0, time_axis.Nt, 1)
    dt = time_axis.dt

    # Get parameters
    c_ref = params["c_ref"]
    pml_rho = params["pml_rho"]
    pml_u = params["pml_u"]

    # Initialize variables
    shape = tuple(list(medium.domain.N) + [
        len(medium.domain.N),
    ])
    shape_one = tuple(list(medium.domain.N) + [
        1,
    ])
    if u0 is None:
        u0 = pml_u.replace_params(jnp.zeros(shape))
    else:
        assert u0.dim == len(medium.domain.N)
    if p0 is None:
        p0 = pml_rho.replace_params(jnp.zeros(shape_one))
    else:
        if settings.smooth_initial:
            p0_params = p0.params[..., 0]
            p0_params = jnp.expand_dims(smooth(p0_params), -1)
            p0 = p0.replace_params(p0_params)

        # Force u(t=0) to be zero accounting for time staggered grid
        u0 = -dt * momentum_conservation_rhs(
            p0, u0, medium, c_ref=c_ref, dt=dt) / 2

    # Initialize acoustic density
    rho = (p0.replace_params(
        jnp.stack([p0.params[..., i]
                   for i in range(p0.domain.ndim)], axis=-1)) / p0.domain.ndim)
    rho = rho / (medium.sound_speed**2)

    # define functions to integrate
    fields = [p0, u0, rho]

    def scan_fun(fields, n):
        p, u, rho = fields
        if sources is None:
            mass_src_field = 0.0
        else:
            mass_src_field = sources.on_grid(n)

        du = momentum_conservation_rhs(p, u, medium, c_ref=c_ref, dt=dt)
        u = pml_u * (pml_u * u + dt * du)

        drho = mass_conservation_rhs(p,
                                     u,
                                     mass_src_field,
                                     medium,
                                     c_ref=c_ref,
                                     dt=dt)
        rho = pml_rho * (pml_rho * rho + dt * drho)

        p = pressure_from_density(rho, medium)
        return [p, u, rho], sensors(p, u, rho)

    if settings.checkpoint:
        scan_fun = jax_checkpoint(scan_fun)

    logger.debug("Starting simulation using generic OnGrid code")
    _, ys = scan(scan_fun, fields, output_steps)

    return ys


def fourier_wave_prop_params(
    medium: Medium[FourierSeries],
    time_axis: TimeAxis,
    *,
    settings: TimeWavePropagationSettings,
    **kwargs,
):
    dt = time_axis.dt

    # Use settings to determine reference sound speed
    c_ref = settings.c_ref(medium)

    # Making PML on grid for rho and u
    def make_pml(staggering=0.0):
        pml_grid = td_pml_on_grid(medium,
                                  dt,
                                  c0=c_ref,
                                  dx=medium.domain.dx[0],
                                  coord_shift=staggering)
        pml = FourierSeries(pml_grid, medium.domain)
        return pml

    pml_rho = make_pml()
    pml_u = make_pml(staggering=0.5)

    # Get k-space operator
    fourier = kspace_op(medium.domain, c_ref, dt)

    return {
        "pml_rho": pml_rho,
        "pml_u": pml_u,
        "fourier": fourier,
        "c_ref": c_ref
    }


@operator(init_params=fourier_wave_prop_params)
def simulate_wave_propagation(
    medium: Medium[FourierSeries],
    time_axis: TimeAxis,
    *,
    settings: TimeWavePropagationSettings = TimeWavePropagationSettings(),
    sources=None,
    sensors=None,
    u0=None,
    p0=None,
    params=None,
):
    r"""Simulates the wave propagation operator using the PSTD method. This
    implementation is equivalent to the `kspaceFirstOrderND` function in the
    k-Wave Toolbox.

    Args:
      medium (FourierOrScalars): The medium.
      time_axis (TimeAxis): The time axis.
      sources (Any): The source terms. It can be any jax traceable object that
        implements the method `sources.on_grid(n)`, which returns the source
        field at the nth time step.
      sensors (Any): The sensor terms. It can be any jax traceable object that
        can be called as `sensors(p,u,rho)`, where `p` is the pressure field,
        `u` is the velocity field, and `rho` is the density field. The return
        value of this function is the recorded field. If `sensors` is not
        specified, the recorded field is the entire pressure field.
      u0 (Field): The initial velocity field. If `None`, the initial velocity
        field is set depending on the `p0` value, such that `u(t=0)=0`. Note that
        the velocity field is staggered forward by half time step relative to the
        pressure field.
      p0 (Field): The initial pressure field. If `None`, the initial pressure
        field is set to zero.
      checkpoint (CheckpointType): The kind of checkpointing to use for the
        simulation. See `CheckpointType` and `ScanCheckpoint` for more details.
      smooth_initial (bool): Whether to smooth the initial conditions.
      params: The operator parameters.

    Returns:
      Any: The recording of the sensors at each time step.
    """
    # Default sensors simply return the presure field
    if sensors is None:
        sensors = lambda p, u, rho: p

    # Setup parameters
    output_steps = jnp.arange(0, time_axis.Nt, 1)
    dt = time_axis.dt

    # Get parameters
    c_ref = params["c_ref"]
    pml_rho = params["pml_rho"]
    pml_u = params["pml_u"]

    # Initialize variables
    shape = tuple(list(medium.domain.N) + [
        len(medium.domain.N),
    ])
    shape_one = tuple(list(medium.domain.N) + [
        1,
    ])
    if u0 is None:
        u0 = pml_u.replace_params(jnp.zeros(shape))
    else:
        assert u0.dim == len(medium.domain.N)
    if p0 is None:
        p0 = pml_rho.replace_params(jnp.zeros(shape_one))
    else:
        if settings.smooth_initial:
            p0_params = p0.params[..., 0]
            p0_params = jnp.expand_dims(smooth(p0_params), -1)
            p0 = p0.replace_params(p0_params)

        # Force u(t=0) to be zero accounting for time staggered grid
        u0 = (-dt * momentum_conservation_rhs(
            p0, u0, medium, c_ref=c_ref, dt=dt, params=params["fourier"]) / 2)

    # Initialize acoustic density
    rho = (p0.replace_params(
        jnp.stack([p0.params[..., i]
                   for i in range(p0.domain.ndim)], axis=-1)) / p0.domain.ndim)
    rho = rho / (medium.sound_speed**2)

    # define functions to integrate
    fields = [p0, u0, rho]

    def scan_fun(fields, n):
        p, u, rho = fields
        if sources is None:
            mass_src_field = 0.0
        else:
            mass_src_field = sources.on_grid(n)

        du = momentum_conservation_rhs(p,
                                       u,
                                       medium,
                                       c_ref=c_ref,
                                       dt=dt,
                                       params=params["fourier"])
        u = pml_u * (pml_u * u + dt * du)

        drho = mass_conservation_rhs(p,
                                     u,
                                     mass_src_field,
                                     medium,
                                     c_ref=c_ref,
                                     dt=dt,
                                     params=params["fourier"])
        rho = pml_rho * (pml_rho * rho + dt * drho)

        p = pressure_from_density(rho, medium)
        return [p, u, rho], sensors(p, u, rho)

    # Define the scanning function according to the checkpoint type
    if settings.checkpoint:
        scan_fun = jax_checkpoint(scan_fun)

    logger.debug("Starting simulation using FourierSeries code")
    _, ys = scan(scan_fun, fields, output_steps)

    return ys


if __name__ == "__main__":
    pass
