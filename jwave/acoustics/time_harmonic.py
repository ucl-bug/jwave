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

from math import factorial
from typing import Union

import jax
from jax import numpy as jnp
from jax.lax import while_loop
from jax.scipy.sparse.linalg import bicgstab, gmres
from jaxdf import operator
from jaxdf.discretization import Field, FourierSeries, OnGrid
from jaxdf.geometry import Domain
from jaxdf.operators import functional
from jaxdf.operators.differential import laplacian
from jaxdf.operators.functions import functional

from jwave.geometry import Medium

from .operators import helmholtz, scale_source_helmholtz


@operator
def angular_spectrum(
    pressure: FourierSeries,
    *,
    z_pos,
    f0,
    medium,
    padding=0,
    angular_restriction=True,
    unpad_output=True,
    params=None,
) -> FourierSeries:
    """Similar to `angularSpectrumCW` from the k-Wave toolbox.

    Projects an input plane of single-frequency
    continuous wave data to the parallel plane specified by z_pos using the
    angular spectrum method. The implementation follows the spectral
    propagator with angular restriction described in reference [1].

    For linear projections in a lossless medium, just the sound speed can
    be specified. For projections in a lossy medium, the parameters are
    given as fields to the input structure medium.

    See [[Zeng and McGhough, 2008](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3408224/)] for
    more details.

    Note: The absorption coefficient is not considered in this function, so the medium is
    assumed to be lossless.


    Args:
        pressure (FourierSeries): omplex pressure values over the input plane $[Pa]$
        z_pos (float): Specifies the relative z-position of the plane of projection.
        f0 (float): The frequency of the input plane.
        medium (Medium): Specifies the speed of sound, density and absorption in the medium.
        padding (Union[str, int], optional): Controls the grid expansion used for
          evaluating the angular spectrum. Defaults to 0.
        angular_restriction (bool, optional): If true, uses the angular restriction method
          specified in [[Zeng and McGhough, 2008](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3408224/)]. Defaults to True.
        unpad_output (bool, optional): If true, the output is padded to the same size as the input. Defaults to True.

    Returns:
        FourierSeries: _description_
    """
    # Get literals
    c0 = medium.sound_speed
    k = 2 * jnp.pi * f0 / c0
    k_t_sq = k**2

    # Pad the input field
    p = pressure.on_grid[..., 0]
    p = jnp.pad(p, padding, mode="constant", constant_values=0)

    # Update the domain
    domain = Domain(
        (p.shape[0], p.shape[1]),    # (Nx, Ny)
        pressure.domain.dx,    # Grid spacing doesn't change
    )
    pressure_padded = FourierSeries(p, domain)

    # Define cutoffs
    freq_grid = pressure_padded._freq_grid
    freq_grid = freq_grid.at[0, 0].set(0.0)
    k_x_sq = jnp.sum(freq_grid**2, axis=-1)
    kz = jnp.sqrt(k_t_sq - k_x_sq + 0j)

    # Evaluate base propagator
    H = jnp.conj(jnp.exp(1j * z_pos * kz))

    # Apply angular restriction, i.e. a hard low-pass filter
    D = min(pressure_padded.domain.size)
    kc = k * jnp.sqrt(0.5 * D**2 / (0.5 * D**2 + z_pos**2))
    H_restrict = jnp.where(k_x_sq <= kc**2, H, 0.0j)
    H = jnp.where(angular_restriction, H_restrict, H)

    # Apply the spectral porpagator
    p_hat = jnp.fft.fftn(pressure_padded.on_grid[..., 0])
    p_hat_plane = p_hat * H
    p_plane = jnp.fft.ifftn(p_hat_plane)

    # Unpad
    if padding > 0 and unpad_output:
        p_plane = p_plane[padding:-padding, padding:-padding]
        p_plane = FourierSeries(p_plane, pressure.domain)
    else:
        p_plane = FourierSeries(p_plane, domain)

    return p_plane


# Building base PML
def _cbs_pml(field: OnGrid,
             k0: object = 1.0,
             pml_size: object = 32,
             alpha: object = 1.0):
    medium = Medium(domain=field.domain, pml_size=pml_size)
    N = 4

    def pml_edge(x):
        return x / 2 - pml_size

    def num(x):
        return (alpha**2) * (N - alpha * x + 2j * k0 * x) * (
            (alpha * x)**(N - 1))

    def den(x):
        return sum([((alpha * x)**i) / float(factorial(i))
                    for i in range(N + 1)]) * factorial(N)

    def transform_fun(x):
        return num(x) / den(x)

    delta_pml = jnp.asarray(list(map(pml_edge, medium.domain.N)))
    coord_grid = Domain(N=medium.domain.N,
                        dx=tuple([1.0] * len(medium.domain.N))).grid
    coord_grid = coord_grid

    diff = jnp.abs(coord_grid) - delta_pml
    diff = jnp.where(diff > 0, diff, 0) / 4

    dist = jnp.sqrt(jnp.sum(diff**2, -1))
    k_k0 = transform_fun(dist)
    k_k0 = jnp.expand_dims(k_k0, -1)
    return k_k0 + k0**2


def _cbs_norm_units(medium, omega, k0, src):
    r"""Converts problem for the Convergent Born Series
    to work in natural units"""
    # Store conversion variables
    domain = medium.domain
    _conversion = {
        "dx": jnp.mean(jnp.asarray(domain.dx)),
        "omega": omega,
    }

    # Set discretization to 1
    dx = tuple(map(lambda x: x / _conversion["dx"], domain.dx))
    domain = Domain(domain.N, dx)

    # set omega to 1
    omega = 1.0

    # Update sound speed
    if issubclass(type(medium.sound_speed), FourierSeries):
        c = medium.sound_speed.params
    else:
        c = medium.sound_speed
    c = c / (_conversion["dx"] * _conversion["omega"])

    # Update fields
    src = FourierSeries(src.on_grid, domain)
    if issubclass(type(medium.sound_speed), FourierSeries):
        c = FourierSeries(c, domain)

    medium = medium.replace("sound_speed", c)

    # Update k0
    k0 = k0 * _conversion["dx"]

    return medium, omega, k0, src, _conversion


def _cbs_unnorm_units(field, conversion):
    domain = field.domain
    dx = tuple(map(lambda x: x * conversion["dx"], domain.dx))
    domain = Domain(domain.N, dx)

    return FourierSeries(field.params, domain)


@operator
def born_series(
    medium: Medium,
    src: FourierSeries,
    *,
    omega=1.0,
    k0: Union[None, float] = None,
    max_iter=1000,
    tol=1e-8,
    alpha=1.0,
    remove_pml=True,
    print_info=False,
    params=None,
) -> FourierSeries:
    r"""Solves the Helmholtz equation using the
    Convergente Born Series (CBS) method described in
    [Osnabrugge et al, 2016](https://doi.org/10.1016/j.jcp.2016.06.034).

    Note that, differently from the implementation of the Helmholtz operator, here the
    PML layer is added __outside__ of the domain, therefore the solution will be valid
    on the entire input domain. This is because the requires a much larger (but weaker)
    PML to converge in a small number of iterations.

    Args:
      medium (Medium): The acoustic medium. Note that any `density` term is ignored, as
        the original paper does not include a density term. The medium is also assumed to
        be lossless (`atteuation` is ignored), as this is not implemented yet.
      src (FourierSeries): The complex source field.
      omega (object): The angular frequency.
      k0 (Union[None, float]): The wavenumber. If None, it is calculated from the medium as
        `k0 = 0.5*(max(k**2) + min(k**2))` [Osnabrugge et al, 2016](https://doi.org/10.1016/j.jcp.2016.06.034). Defaults to None.
      max_iter (object): The maximum number of iterations.
      tol (object): The relative tolerance for the convergence.
      alpha (object): The amplitude parameter of the PML. See Appendix of [Osnabrugge et al, 2016](https://doi.org/10.1016/j.jcp.2016.06.034)
      remove_pml (bool): If true, the PML is removed from the solution. Defaults to True.
      print_info (bool): If true, prints information about the convergence. Defaults to False.

    Returns:
      FourierSeries: The complex solution field.
    """

    # TODO: Implement absorption term

    # Support functions
    def enlarge_doimain(domain, pml_size):
        new_N = tuple([x + 2 * pml_size for x in domain.N])
        return Domain(new_N, domain.dx)

    def pad_fun(u):
        pad_size = tuple([(pml_size, pml_size)
                          for _ in range(len(u.domain.N))] + [(0, 0)])
        return FourierSeries(jnp.pad(u.on_grid, pad_size),
                             enlarge_doimain(u.domain, pml_size))

    def cbs_helmholtz(field, k_sq):
        return laplacian(field) + k_sq * field

    # Define k0 if not given
    if k0 is None:
        k_max = omega / functional(medium.sound_speed)(jnp.amax)
        k_min = omega / functional(medium.sound_speed)(jnp.amin)
        k0 = jnp.sqrt(0.5 * (k_max**2 + k_min**2))

    # Work in normalized units
    medium, omega, k0, src, _conversion = _cbs_norm_units(
        medium, omega, k0, src)

    # Constants
    pml_size = medium.pml_size
    sound_speed = medium.sound_speed

    # Padding the field
    src = scale_source_helmholtz(src, medium)
    src = pad_fun(src)
    norm_initial = jnp.linalg.norm(src.on_grid)

    # Constructing heterogeneous k^2
    _sos = sound_speed.on_grid if type(
        sound_speed) is FourierSeries else sound_speed
    k_biggest = jnp.amax((omega / _sos))
    k_sq = _cbs_pml(src, k_biggest, pml_size, alpha)
    k_sq = k_sq.at[pml_size:-pml_size,
                   pml_size:-pml_size].set(((omega / _sos)**2) + 0j)
    k_sq = FourierSeries(k_sq, src.domain)

    # Finding stable epsilon
    epsilon = jnp.amax(jnp.abs((k_sq.on_grid - k0**2)))

    # Setting guess
    guess = FourierSeries.empty(src.domain) + 0j

    # Setting up loop
    carry = (0, guess)

    def resid_fun(field):
        return cbs_helmholtz(field, k_sq) + src

    def cond_fun(carry):
        numiter, field = carry
        cond_1 = numiter < max_iter
        cond_2 = jnp.linalg.norm(resid_fun(field).on_grid) / norm_initial > tol
        return cond_1 * cond_2

    def body_fun(carry):
        numiter, field = carry
        field = born_iteration(field, k_sq, src, k0=k0, epsilon=epsilon)
        return numiter + 1, field

    numiters, out_field = while_loop(cond_fun, body_fun, carry)

    if print_info:
        v = jnp.linalg.norm(resid_fun(out_field).on_grid) / norm_initial
        jax.debug.print("Converged in {c} iterations. Relative error: {v}",
                        c=numiters,
                        v=v)

    # Remove padding
    # TODO: Remove this switch-like statement to support all dimensions
    num_dims = len(out_field.domain.N)
    if remove_pml:
        if num_dims == 1:
            _out_field = out_field.on_grid[pml_size:-pml_size]
        elif num_dims == 2:
            _out_field = out_field.on_grid[pml_size:-pml_size,
                                           pml_size:-pml_size]
        elif num_dims == 3:
            _out_field = out_field.on_grid[pml_size:-pml_size,
                                           pml_size:-pml_size,
                                           pml_size:-pml_size]
        else:
            raise ValueError("Only 1, 2, or 3 dimensions are supported.")
    else:
        _out_field = out_field.on_grid

    # Rescale field according to omega
    out_field = -1j * omega * FourierSeries(_out_field, medium.domain)

    out_field = _cbs_unnorm_units(out_field, _conversion)

    return out_field


@operator
def born_iteration(field: Field,
                   k_sq: Field,
                   src: Field,
                   *,
                   k0,
                   epsilon,
                   params=None) -> FourierSeries:
    r"""Implements one step of the Convergente Born Series (CBS) method.

    $$
    u_{k+1} = u_k - \gamma\left[u_k - G(Vu_k + s)\right]
    $$

    where $\gamma = -(i/\varepsilon)V$. Here $V$ and $G$ are implemented by the
    `scattering_potential` and `homogeneous_helmholtz_green` operators.

    Args:
      field (FourierSeries): The current field $u_k$.
      k_sq (FourierSeries): The heterogeneous wavenumber squared.
      src (FourierSeries): The complex source field $s$.
      k0 (object): The wavenumber.
      epsilon (object): The absorption of the preconditioner.

    Returns:
      FourierSeries: The updated field $u_{k+1}$.

    """
    V1 = scattering_potential(field, k_sq, k0=k0, epsilon=epsilon)
    G = homogeneous_helmholtz_green(V1 + src, k0=k0, epsilon=epsilon)
    V2 = scattering_potential(field - G, k_sq, k0=k0, epsilon=epsilon)

    return field - (1j / epsilon) * V2


@operator
def scattering_potential(field: Field,
                         k_sq: Field,
                         *,
                         k0=1.0,
                         epsilon=0.1,
                         params=None) -> Field:
    r"""Implements the scattering potential of the CBS method.

    Args:
      field (FourierSeries): The current field $u$.
      k_sq (FourierSeries): The heterogeneous wavenumber squared.
      k0 (object): The wavenumber.
      epsilon (object): The absorption parameter.

    Returns:
      FourierSeries: The scattering potential.
    """

    k = k_sq - k0**2 - 1j * epsilon
    out = field * k
    return out


@operator
def homogeneous_helmholtz_green(field: FourierSeries,
                                *,
                                k0=1.0,
                                epsilon=0.1,
                                params=None):
    r"""Implements the Green's operator for the homogeneous Helmholtz equation.

    Note that being the field a `FourierSeries`, the Green's function is periodic.

    Args:
      field (FourierSeries): The input field $u$.
      k0 (object): The wavenumber.
      epsilon (object): The absorption parameter.

    Returns:
      FourierSeries: The result of the Green's operator on $u$.
    """
    freq_grid = field._freq_grid
    p_sq = jnp.sum(freq_grid**2, -1)

    g_fourier = 1.0 / (p_sq - (k0**2) - 1j * epsilon)
    u = field.on_grid[..., 0]
    u_fft = jnp.fft.fftn(u)
    Gu_fft = g_fourier * u_fft
    Gu = jnp.fft.ifftn(Gu_fft)
    return field.replace_params(Gu)


@operator
def rayleigh_integral(
    pressure: FourierSeries,
    *,
    r,
    f0,
    sound_speed=1500.0,
    params=None,
):
    """
    Rayleigh integral for a `FourierSeries` field.

    Args:
      pressure (FourierSeries): pressure field, corresponding to $u$ on the plane.
      r (jnp.ndarray): distance from the origin of the pressure plane.
        Must be a 3D array.
      f0 (float): frequency of the source.
      sound_speed (float): Value of the homogeneous sound speed where
        the rayleigh integral is computed. Default is 1500 m/s.

    Returns:
      complex64: Rayleigh integral at `r`
    """
    # TODO: Override vmap on second dimension using Fourier implementation of the rayleigh integral
    #       instead of the direct implementation. See Appendix B of
    #       https://asa.scitation.org/doi/pdf/10.1121/1.4928396 for details.

    # Checks
    if pressure.ndim != 2:
        raise ValueError("Only 2D domains are supported.")

    assert r.shape == (3, ), "The target position must be a 3D vector."

    # Terms in the Rayleigh integral
    # See eq. A2 and A3 in https://asa.scitation.org/doi/10.1121/1.4928396
    k = 2 * jnp.pi * f0 / sound_speed

    def exp_term(x, y, z):
        """The exponential term of the Rayleigh integral
        (first kind). This is basically the Green's function
        of a dirac delta with Sommerfield radiation conditions."""
        r = jnp.sqrt(x**2 + y**2 + z**2)
        return jnp.exp(1j * k * r) / r

    def direc_exp_term(x, y, z):
        """Derivative of the exponential term in the Rayleigh integral,
          along the z-axis (second kind). This is basically the Green's function
          of a dipole oriented along the z-axis.
        p = pressure.on_grid[...,0]
        """
        _, direc_derivative = jax.jvp(exp_term, (x, y, z), (0.0, 0.0, 1.0))
        return direc_derivative

    # Integral calculation as a finite sum
    area = pressure.domain.cell_volume
    plane_grid = pressure.domain.grid

    # Append z dimension of zeros to the last dimension
    z_dim = jnp.zeros(plane_grid.shape[:-1] + (1, ))
    plane_grid = jnp.concatenate((plane_grid, z_dim), axis=-1)

    # Distance from r to the plane
    R = jnp.abs(r - plane_grid)

    # Weights of the Rayleigh integral
    weights = jax.vmap(jax.vmap(direc_exp_term, in_axes=(0, 0, 0)),
                       in_axes=(0, 0, 0))(R[..., 0], R[..., 1], R[..., 2])
    return jnp.sum(weights * pressure.on_grid) * area


@operator
def helmholtz_solver(
    medium: Medium,
    omega: object,
    source: OnGrid,
    *,
    guess: Union[OnGrid, None] = None,
    method: str = "gmres",
    checkpoint: bool = True,
    params=None,
    **kwargs,
):
    """

    Args:
        medium (Medium): _description_
        omega (object): _description_
        source (OnGrid): _description_
        guess (Union[OnGrid, None], optional): _description_. Defaults to None.
        method (str, optional): _description_. Defaults to 'gmres'.
        checkpoint (bool, optional): _description_. Defaults to True.
        params (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    source = scale_source_helmholtz(source, medium)

    if params is None:
        params = helmholtz.default_params(source, medium, omega=omega)

    def helm_func(u):
        return helmholtz(u, medium, omega=omega, params=params)

    if checkpoint:
        helm_func = jax.checkpoint(helm_func)

    if guess is None:
        guess = source * 0

    tol = kwargs["tol"] if "tol" in kwargs else 1e-3
    restart = kwargs["restart"] if "restart" in kwargs else 10
    maxiter = kwargs["maxiter"] if "maxiter" in kwargs else 1000
    solve_method = kwargs[
        "solve_method"] if "solve_method" in kwargs else "batched"
    if method == "gmres":
        out = gmres(
            helm_func,
            source,
            guess,
            tol=tol,
            restart=restart,
            maxiter=maxiter,
            solve_method=solve_method,
        )[0]
    elif method == "bicgstab":
        out = bicgstab(helm_func, source, guess, tol=tol, maxiter=maxiter)[0]
    return -1j * omega * out


def helmholtz_solver_verbose(
    medium: Medium,
    omega: float,
    source: OnGrid,
    guess: Union[OnGrid, None] = None,
    params=None,
    **kwargs,
):

    src_magn = jnp.linalg.norm(source.on_grid)
    source = source / src_magn

    tol = kwargs["tol"] if "tol" in kwargs else 1e-3
    residual_magnitude = jnp.linalg.norm(
        helmholtz(source, medium, omega).params)
    maxiter = kwargs["maxiter"] if "maxiter" in kwargs else 1000

    if params is None:
        params = helmholtz(source, medium, omega)._op_params

    if guess is None:
        guess = source * 0

    kwargs["maxiter"] = 1
    kwargs["tol"] = 0.0
    iterations = 0

    if isinstance(medium.sound_speed, Field):
        min_sos = functional(medium.sound_speed)(jnp.amin)
    else:
        min_sos = jnp.amin(medium.sound_speed)

    @jax.jit
    def solver(medium, guess, source):
        guess = helmholtz_solver(medium,
                                 omega,
                                 source,
                                 guess,
                                 "gmres",
                                 **kwargs,
                                 params=params)
        residual = helmholtz(guess, medium, omega, params=params) - source
        residual_magnitude = jnp.linalg.norm(residual.params)
        return guess, residual_magnitude

    while residual_magnitude > tol and iterations < maxiter:
        guess, residual_magnitude = solver(medium, guess, source)

        iterations += 1

        # Print iteration info
        print(
            f"Iteration {iterations}: residual magnitude = {residual_magnitude:.4e}, tol = {tol:.2e}",
            flush=True,
        )

    return -1j * omega * guess * src_magn * 2 / (source.domain.dx[0] * min_sos)
