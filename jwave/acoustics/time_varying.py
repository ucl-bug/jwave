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

import warnings
from typing import Optional, TypeVar, Union

import numpy as np
from diffrax import (AbstractAdjoint, AbstractStepSizeController,
                     ConstantStepSize, DirectAdjoint, ODETerm,
                     RecursiveCheckpointAdjoint, SaveAt, diffeqsolve)
from diffrax.custom_types import Scalar
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxdf import Field, operator
from jaxdf.discretization import FourierSeries, OnGrid
from jaxdf.operators import functional, shift_operator, sum_over_dims

from jwave.acoustics.spectral import kspace_op
from jwave.geometry import (Medium, MediumAllScalars, MediumOnGrid, Sensors,
                            Sources)
from jwave.ode import SemiImplicitEulerCorrected, TimeAxis
from jwave.signal_processing import smooth

from .pml import td_pml_on_grid

Any = TypeVar("Any")


def _shift_rho(rho0, direction, dx):
    if isinstance(rho0, OnGrid):
        rho0_params = rho0.params[..., 0]

        def linear_interp(u, axis):
            return 0.5 * (jnp.roll(u, -direction, axis) + u)

        rho0 = jnp.stack(
            [linear_interp(rho0_params, n) for n in range(rho0.ndim)], axis=-1)
    elif isinstance(rho0, Field):
        rho0 = shift_operator(rho0, direction * dx)
    else:
        pass
    return rho0


# Settings for the wave solver
@register_pytree_node_class
class WaveSolverSettings:

    def __init__(self,
                 PMLAlpha: Optional[Scalar] = 2.0,
                 PMLSize: Optional[Scalar] = 20,
                 PMLInside: Optional[bool] = True,
                 SmoothP0: Optional[bool] = True,
                 SmoothSoundSpeed: Optional[bool] = False,
                 SmoothDensity: Optional[bool] = False):
        """
        Initializes the settings for a Wave Solver.

        Args:
            PMLAlpha (Scalar, optional): Alpha value for Perfectly
                Matched Layer (PML). Defaults to 2.0.
            PMLSize (Scalar, optional): Size of the Perfectly Matched Layer (PML).
                Defaults to 20.
            PMLInside (bool, optional): Flag indicating whether PML is inside.
                Defaults to True.
            SmoothP0 (bool, optional): Flag for smoothing initial pressure distribution.
                Defaults to True.
            SmoothSoundSpeed (bool, optional): Flag for smoothing sound speed.
                Defaults to False.
            SmoothDensity (bool, optional): Flag for smoothing density. Defaults to False.
        """
        self.PMLAlpha = PMLAlpha
        self.PMLSize = PMLSize
        self.PMLInside = PMLInside
        self.SmoothP0 = SmoothP0
        self.SmoothSoundSpeed = SmoothSoundSpeed
        self.SmoothDensity = SmoothDensity

    def tree_flatten(self):
        children = (self.PMLAlpha, )
        aux = (self.PMLSize, self.PMLInside, self.SmoothP0,
               self.SmoothSoundSpeed, self.SmoothDensity)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)

    def __repr__(self) -> str:
        """
        Generates a machine-readable string representation of the instance.

        Returns:
            str: String representation of the instance.
        """
        return (
            f"WaveSolverSettings(PMLAlpha={self.PMLAlpha}, PMLSize={self.PMLSize}, "
            f"PMLInside={self.PMLInside}, SmoothP0={self.SmoothP0}, "
            f"SmoothSoundSpeed={self.SmoothSoundSpeed}, SmoothDensity={self.SmoothDensity})"
        )

    def __str__(self) -> str:
        """
        Generates a human-readable string representation of the instance.

        Returns:
            str: String representation of the instance.
        """
        return (
            f"WaveSolverSettings with PMLAlpha: {self.PMLAlpha}, PMLSize: {self.PMLSize}, "
            f"PMLInside: {self.PMLInside}, SmoothP0: {self.SmoothP0}, "
            f"SmoothSoundSpeed: {self.SmoothSoundSpeed}, SmoothDensity: {self.SmoothDensity}"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, WaveSolverSettings):
            return False
        return (self.PMLAlpha == other.PMLAlpha
                and self.PMLSize == other.PMLSize
                and self.PMLInside == other.PMLInside
                and self.SmoothP0 == other.SmoothP0
                and self.SmoothSoundSpeed == other.SmoothSoundSpeed
                and self.SmoothDensity == other.SmoothDensity)


@operator
def pressure_from_density(rho: Field, medium: Medium, *, params=None) -> Field:
    r"""Compute the pressure field from the density field.

    Args:
      rho (Field): The density field.
      medium (Medium): The medium.
      params: The operator parameters. **Unused**

    Returns:
      Field: The pressure field.
    """
    rho_sum = sum_over_dims(rho)
    c0 = medium.sound_speed
    return (c0**2) * rho_sum, params


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

    dp = jnp.stack([single_grad(i) for i in range(p.ndim)], axis=-1)
    update = -p.replace_params(dp) / rho0

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

    du = jnp.stack([single_grad(i, u.params[..., i]) for i in range(p.ndim)],
                   axis=-1)
    update = -p.replace_params(du) * rho0 + 2 * mass_source / (c0 * p.ndim *
                                                               dx)

    return update, params


def _rewrite_saveat(
    medium: Medium,
    saveat: Union[SaveAt, None],
    time_axis: TimeAxis,
    sensors: Sensors,
):
    # TODO: Deal properly with sub-saveat. There is some redundancy between
    # SaveAt and sensors. Most likely, sensors should be some kind of
    # derived class of SaveAt.

    # If saveat is None, generate a default one that samples at the
    # time steps of the time axis.
    if saveat is None:
        ts = time_axis.to_array()
        saveat = SaveAt(t0=True, t1=True, ts=ts, steps=False)

    # Warn the user if subs is set and also sensors are set,
    # since it will be overwritten.
    if saveat.subs.fn is not None and Sensors is not None:
        warnings.warn(
            f"Both sensors and saveat.fn are set. The latter will be overwritten"
        )

    # Define default sampling function, that only returns the pressure
    # if sensors is None and saveat.fn is None
    if sensors is None:
        sensors = lambda p, u, rho: p

    if sensors is not None:
        # Define the sampling function
        print("Redefining sampling func")

        def sampling(t, fields, args):
            u, rho = fields

            # Generate p
            p = pressure_from_density(rho, medium)

            return sensors(p, u, rho)

        return SaveAt(
            t0=saveat.subs.t0,
            t1=saveat.subs.t1,
            ts=saveat.subs.ts,
            steps=saveat.subs.steps,
            fn=sampling,
            dense=saveat.dense,
            solver_state=saveat.solver_state,
            controller_state=saveat.controller_state,
            made_jump=saveat.made_jump,
        )

    print("I should not be here")
    return saveat


# General operator
@operator
def simulate_wave_propagation(
        medium: MediumOnGrid,
        time_axis: TimeAxis,
        solver: SemiImplicitEulerCorrected,
        *,
        sources: Optional[Sources] = None,
        sensors: Optional[Sensors] = None,
        u0: Optional[OnGrid] = None,
        p0: Optional[OnGrid] = None,
        settings: Optional[WaveSolverSettings] = WaveSolverSettings(),
        adjoint: Optional[AbstractAdjoint] = None,
        saveat: Optional[SaveAt] = SaveAt(steps=True),
        stepsize_controller: Optional[
            AbstractStepSizeController] = ConstantStepSize(),
        max_steps: Optional[int] = None,
        params=None):
    # Must define the maximum number of steps, otherwise DirectAdjoint will not
    # allow for backpropagation (but only forward AD)
    if max_steps is None and adjoint is DirectAdjoint():
        max_steps = time_axis.Nt

    raise NotImplementedError("The general operator is not implemented yet")


def simulate_wave_propagation(*args, **kwargs):
    warnings.warn(
        "The `simulate_wave_propagation` operator is deprecated. Use the acoustic_solver instead."
    )
    return acoustic_solver(*args, **kwargs)


def kspace_acoustic_solver_params(
    medium: Union[MediumAllScalars, MediumOnGrid],
    time_axis: TimeAxis,
    solver: SemiImplicitEulerCorrected,
    *args,
    **kwargs,
):
    dt = time_axis.dt
    c_ref = functional(medium.sound_speed)(jnp.amax)

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
        "c_ref": c_ref,
    }


def _initialize_fourier_fields(
        medium: Union[MediumAllScalars, MediumOnGrid],
        time_axis: TimeAxis,
        sample_field: FourierSeries,
        params: Any,
        u0: Optional[FourierSeries] = None,
        p0: Optional[FourierSeries] = None,
        settings: Optional[WaveSolverSettings] = WaveSolverSettings(),
):
    shape = tuple(list(medium.domain.N) + [
        len(medium.domain.N),
    ])
    shape_one = tuple(list(medium.domain.N) + [
        1,
    ])
    c_ref = params["c_ref"]
    dt = time_axis.dt

    # Initialize u0 and p0 to zero if not given
    if p0 is None:
        p0 = sample_field.replace_params(jnp.zeros(shape_one))
    else:
        if settings.SmoothP0:
            p0_params = p0.params[..., 0]
            p0_params = jnp.expand_dims(smooth(p0_params), -1)
            p0 = p0.replace_params(p0_params)

        # Force u(t=0) to be zero accounting for time staggered grid, if not set
        if u0 is None:
            u0 = 0.0 * p0    #TODO: This should not be necessary
            u0 = (-(dt / 2) * momentum_conservation_rhs(
                p0, u0, medium, c_ref=c_ref, dt=dt, params=params["fourier"]))

    # Define initial acoustic density
    rho0 = (p0.replace_params(
        jnp.stack([p0.params[..., i]
                   for i in range(p0.ndim)], axis=-1)) / p0.ndim)
    rho0 = rho0 / (medium.sound_speed**2)

    # Smooth sound speed and density if requested
    if settings.SmoothSoundSpeed:
        sos_params = medium.sound_speed.params[..., 0]
        sos_params = jnp.expand_dims(smooth(sos_params), -1)
        medium.sound_speed = medium.sound_speed.replace_params(sos_params)

    if settings.SmoothDensity:
        rho_params = medium.density.params[..., 0]
        rho_params = jnp.expand_dims(smooth(rho_params), -1)
        medium.density = medium.density.replace_params(rho_params)

    return medium, u0, rho0


# The default solver is equivalent to k-wave, and uses the k-space correction. The
# k-space correction is included in the gradient operators
@operator(init_params=kspace_acoustic_solver_params)
def acoustic_solver(
    medium: Union[MediumAllScalars, MediumOnGrid],
    time_axis: TimeAxis,
    solver: SemiImplicitEulerCorrected = SemiImplicitEulerCorrected(),
    *,
    sources: Optional[Sources] = None,
    sensors: Optional[Sensors] = None,
    u0: Optional[OnGrid] = None,
    p0: Optional[OnGrid] = None,
    settings: Optional[WaveSolverSettings] = WaveSolverSettings(),
    adjoint: Optional[AbstractAdjoint] = None,
    saveat: Optional[SaveAt] = None,
    stepsize_controller: Optional[
        AbstractStepSizeController] = ConstantStepSize(),
    max_steps: Optional[int] = None,
    params=None,
):
    # With the modified k-space correction, the stepsize_controller can only
    # be ConstantStepSize. Raise an error if it is not
    if not isinstance(stepsize_controller, ConstantStepSize):
        raise NotImplementedError(
            "The modified k-space correction only works with a constant stepsize controller"
        )

    # Rewrite the saveat
    saveat = _rewrite_saveat(medium, saveat, time_axis, sensors)

    # If saving at steps, then max_steps must be set
    if saveat.subs.steps and max_steps is None:
        raise ValueError("max_steps must be set when saveat.steps is True")

    # Get operator parameters
    c_ref = params["c_ref"]
    pml_rho = params["pml_rho"]
    pml_u = params["pml_u"]
    fourier = params["fourier"]

    # Make initial conditions
    medium, u0, rho0 = _initialize_fourier_fields(medium, time_axis, pml_u,
                                                  params, u0, p0, settings)
    fields = (u0, rho0)

    # Define the functions for the symplectic Euler integration
    dt = time_axis.dt

    def f(t, rho, args):
        # Generate p
        p = pressure_from_density(rho, medium)
        return momentum_conservation_rhs(p,
                                         rho,
                                         medium,
                                         c_ref=c_ref,
                                         dt=dt,
                                         params=params["fourier"])

    def g(t, u, args):
        # Generate (a fake) p
        # TODO: This should not be needed here. In general, p should not
        # exist in those equations. However, the current implementation of
        # the pair of ode functions requires it.
        p = pressure_from_density(u, medium)

        # Generate source field if needed
        n = jnp.round(t * dt)
        if sources is None:
            mass_src_field = 0.0
        else:
            mass_src_field = sources.on_grid(n)

        return mass_conservation_rhs(p,
                                     u,
                                     mass_src_field,
                                     medium,
                                     c_ref=c_ref,
                                     dt=dt,
                                     params=params["fourier"])

    # Setup diffrax solver
    terms = ODETerm(f), ODETerm(g)
    args = (pml_u, pml_rho, None)

    # Define adjoint
    if adjoint is None:
        n_steps = time_axis.to_array(keep_last=True).shape[0]
        adjoint = RecursiveCheckpointAdjoint(checkpoints=n_steps)

    # Integrate
    solution = diffeqsolve(terms,
                           solver,
                           t0=time_axis.t0,
                           t1=time_axis.t1,
                           dt0=time_axis.dt,
                           y0=fields,
                           args=args,
                           saveat=saveat,
                           stepsize_controller=stepsize_controller,
                           adjoint=adjoint,
                           max_steps=max_steps)

    # TODO: May be interesting to return the full solution, to use the
    # interpolation ability of the solution object, as well as the possibility to integrate
    # against the integration extrema and to take numerical derivatives of the solution
    return solution.ys
