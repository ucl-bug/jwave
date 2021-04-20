from jwave import geometry, signal_processing, spectral, ode
from jax import lax
from jax import numpy as jnp
from jax.scipy.sparse.linalg import gmres, bicgstab
from jax.tree_util import tree_map
from typing import Tuple, Callable
from numpy import arange as nparange
from time import time

from functools import partial
import jax


def get_time_domain_pml(
    PML_size: int, alpha_max: float, dimensions: Tuple[int], power: float = 4.0
) -> jnp.ndarray:
    r"""Returns the absorption coefficient $`\alpha`$ for a PML layer
    to be used in time-domain simulations

    Args:
        PML_size (int): Size of the pml in grid-points
        alpha_max (float):
        dimensions (Tuple[int]):
        power (float, optional):

    Returns:
        jnp.ndarray: The absorption coefficient

    !!! warning
        This function can be included in a computation to be differentiated, but its
        gradient is set to zero. That is, you can't take gradients with
        respect to the PML parameters.

        That's often the intended behaviour, if one's interested
        in the wave physics: the PML is just a tool used to enforce absorbing boundary
        conditions.

    !!! example
        ```python
        grid = kGrid.make_grid(N=(64,64), dx=(0.3, 0.3))
        pml_alpha = get_time_domain_pml(PML_size=10, alpha_max=5, dimensions=grid.N)
        ```
    """
    axis = [signal_processing._dist_from_ends(x) for x in dimensions]
    coords = jnp.stack(jnp.meshgrid(*axis))
    src_axis = list(range(1, len(dimensions) + 1))
    dest_axis = list(reversed(src_axis))
    new_coords = jnp.moveaxis(coords, src_axis, dest_axis)
    alpha = alpha_max * jnp.where(
        new_coords <= PML_size, (1 - new_coords / PML_size) ** power, 0
    )
    return lax.stop_gradient(alpha)


def laplacian_with_pml(
    grid: geometry.kGrid, medium: geometry.Medium, omega: float, sigma_max: float = 2.0
) -> Callable:
    r"""Returns a laplacian operator augmented with a PML. For
    boundary value problems such as the Helmholtz equation.

    The components of the modified gradient are givem by[^1]

    ```math
    \frac{\partial}{\partial x_j} \to \frac{1}{\gamma_j} \frac{\partial}{\partial x_j},
    \qquad
    \gamma_j = 1 + \frac{1}{k_0}\sigma_j(x), \qquad \sigma_j(x)\frac{1}{\|x_j - x_j^{PML}\|}
    ```

    [^1]: http://oomph-lib.maths.man.ac.uk/doc/pml_helmholtz/scattering/latex/refman.pdf

    Args:
        grid (geometry.kGrid):
        medium (geometry.Medium):
        omega (float): [description]
        sigma_max (float, optional): [description]

    Returns:
        Callable: [description]

    !!! example
        ```python
        random_seed = jax.random.PRNGKey(42)

        # Generate laplacian
        grid = kGrid.make_grid(N=(64,64), dx=(0.3, 0.3))
        laplacian = laplacian_with_pml(grid, medium, 1.0)

        # Apply laplacian
        field = jnp.random.normal(random_seed, grid.N)
        L_field = laplacian(field)
        ```
    """
    # Building PML gamma function
    gamma, gamma_prime = _get_gamma_functions(grid, medium, omega, sigma_max)

    def derivative(field, ax, order):
        return spectral.derivative(
            field, grid, 0, ax, degree=order
        )  # 0 is for "unstaggered"

    def lapl(field):
        axis = list(range(field.ndim))

        def deriv(ax):
            return derivative(field, ax, 2) / (gamma[ax] ** 2) - gamma_prime[
                ax
            ] * derivative(field, ax, 1) / (gamma[ax] ** 3)

        return jnp.sum(jnp.stack([deriv(ax) for ax in axis]), axis=0)

    return jax.jit(lapl)


def heterogeneous_laplacian(
    grid: geometry.kGrid,
    medium: geometry.Medium,
    omega: float,
    sigma_max: float = 1.0,
) -> Callable:
    r"""Returns an heterogeneous laplacian operator augmented with a PML:

    ```math
    \hat \nabla_c^2 = \hat \nabla \cdot(c \hat \nabla)
    ```

    For boundary value problems such as the Helmholtz equation.

    Args:
        grid (geometry.kGrid):
        medium (geometry.Medium):
        omega (float): The temporal pulsation
        cmap (jnp.ndarray): The heterogeneous map $`c`$
        sigma_max (float, optional): [description]

    Returns:
        Callable: [description]

    !!! example
        ```python
        random_seed = jax.random.PRNGKey(42)

        # Generate laplacian
        grid = kGrid.make_grid(N=(64,64), dx=(0.3, 0.3))
        c = jnp.ones(N)
        c = c.at[32:].set(2.0)
        laplacian = heterogeneous_laplacian(grid, medium, 1.0, c)

        # Apply laplacian
        field = jnp.random.normal(random_seed, grid.N)
        L_field = laplacian(field)
        ```
    """
    gamma, _ = _get_gamma_functions(grid, medium, omega, sigma_max)

    def derivative(field, ax):
        return spectral.derivative(
            field, grid, 0, ax, degree=1
        )  # 0 is for "unstaggered"

    def lapl(field, cmap):
        axis = list(range(field.ndim))

        def deriv(ax):
            return (
                derivative(cmap * (derivative(field, ax) / gamma[ax]), ax) / gamma[ax]
            )

        return jnp.sum(jnp.stack([deriv(ax) for ax in axis]), axis=0)

    return jax.jit(lapl)


def generalized_laplacian(
    grid: geometry.kGrid,
    medium: geometry.Medium,
    omega: float,
    sigma_max: float = 1.0,
) -> Callable:
    r"""Returns an heterogeneous laplacian operator augmented with a PML:

    ```math
    \hat \nabla_c^2 = \hat \nabla \cdot(c \hat \nabla)
    ```

    For boundary value problems such as the Helmholtz equation.

    Args:
        grid (geometry.kGrid):
        medium (geometry.Medium):
        omega (float): The temporal pulsation
        cmap (jnp.ndarray): The heterogeneous map $`c`$
        sigma_max (float, optional): [description]

    Returns:
        Callable: [description]

    !!! example
        ```python
        random_seed = jax.random.PRNGKey(42)

        # Generate laplacian
        grid = kGrid.make_grid(N=(64,64), dx=(0.3, 0.3))
        c = jnp.ones(N)
        c = c.at[32:].set(2.0)
        laplacian = heterogeneous_laplacian(grid, medium, 1.0, c)

        # Apply laplacian
        field = jnp.random.normal(random_seed, grid.N)
        L_field = laplacian(field)
        ```
    """
    gamma, _ = _get_gamma_functions(grid, medium, omega, sigma_max)

    def derivative(field, ax):
        # 0 is for "unstaggered"
        return spectral.derivative(field, grid, 0, ax, degree=1) / gamma[ax]

    def v_derivative_axis(field, axis):
        return jnp.stack([derivative(field, ax) for ax in axis])

    def v_derivative(field, axis):
        return jnp.stack([derivative(field[ax], ax) for ax in axis])

    def lapl(field, rho0, tau, omega):
        axis = list(range(field.ndim))
        inv_rho_nabla_p = v_derivative_axis(field, axis)
        rho_nabla = rho0 * jnp.sum(v_derivative(inv_rho_nabla_p, axis), axis=0)
        nabla_rho = jnp.sum(v_derivative_axis(rho0, axis) * inv_rho_nabla_p, axis=0)
        return rho_nabla + 1j * omega * tau * (rho_nabla + nabla_rho)

    return jax.jit(lapl)


def _get_gamma_functions(
    grid: geometry.kGrid, medium: geometry.Medium, omega: float, sigma_max: float = 1.0
):
    r"""Returns the $`\gamma`$ functions for the PML.

    !!! warning
        This function can be included in a computation to be differentiated, but its
        gradient is set to zero. That is, you can't take gradients with
        respect to the PML parameters.

        That's often the intended behaviour, as one's often interested
        in the wave physics and the PML is just a tool used to enforce absorbing boundary
        conditions. This wont work if you are interested in the PML layer
        itself (in that case may be wiser to write the PML function from scratch).

    Args:
        grid (geometry.kGrid): [description]
        medium (geometry.Medium): [description]
        omega (float): [description]
        sigma_max (float, optional): [description]. Defaults to 1.0.

    Returns:
        [type]: [description]
    """
    # Building PML gamma function
    axis = [
        jnp.linspace(-n / 2 + dx / 2, n / 2 - dx / 2, n).astype(jnp.float32) * dx
        for n, dx in zip(grid.N, grid.dx)
    ]
    axis = jnp.meshgrid(*axis, indexing="ij")
    pml_size = jnp.stack(
        [x * medium.pml_size * jnp.ones_like(ax) for x, ax in zip(grid.dx, axis)]
    )
    L = jnp.stack(
        [L * dx * jnp.ones_like(ax) / 2 for L, dx, ax in zip(grid.N, grid.dx, axis)]
    )
    axis = jnp.stack(axis)

    # Wavevector in the pml
    k = grid.dx[0] * omega / jnp.amin(medium.sound_speed)

    def gamma_fun_img_part(x, L, pml_size, omega):
        x = abs(x)
        return jax.lax.cond(
            x >= L - pml_size,
            lambda x: sigma_max * ((1 + (x - L) / pml_size) ** 2) / k,
            lambda x: 0.0,
            operand=x,
        )

    gamma_prime_img_part = jax.grad(gamma_fun_img_part)
    gamma = 1 + 1j * jnp.vectorize(gamma_fun_img_part)(axis, L, pml_size, omega)
    gamma_prime = 1j * jnp.vectorize(gamma_prime_img_part)(axis, L, pml_size, omega)
    return lax.stop_gradient((gamma, gamma_prime))  # Non differentiabl


def get_helmholtz_operator(grid: geometry.kGrid, medium: geometry.Medium, omega: float):
    r"""Returns the standard Helmholtz operator that evaluates

    ```math
    \left( \nabla^2 + \frac{\omega^2}{c_0^2}\right)\hat p
    ```

    with a laplacian augmented with a PML layer.

    Args:
        grid (geometry.kGrid): [description]
        medium (geometry.Medium): [description]
        omega (float): [description]

    Returns:
        Callable: The laplacian operator. Requires the inputs `field`,
            `omega` and `medium`.

    !!! example
        ```python
        N = (512, 512); dx = (0.5, 0.5);
        grid = kGrid.make_grid(N, dx)
        medium = Medium(jnp.ones(N), 1., 0., 15)
        omega = 1.
        H = get_helmholtz_operator(grid, medium, omega)

        # Calling the Helmholtz operator
        field = jnp.zeros(N).astype(jnp.complex64)
        field = field.at[32, 32].set(10.0)
        residual = H(field, omega, medium)

        # The operator is linear with respect to the first
        # input
        linear_op = partial(residual, medium=medium, omega=omega)
        residual = linear_op(field)
        ```
    """
    laplacian = laplacian_with_pml(grid, medium, omega)

    def helmholtz_operator(x, omega, medium):
        return laplacian(x) + x * ((omega / medium.sound_speed) ** 2)

    return helmholtz_operator


def get_helmholtz_operator_density(
    grid: geometry.kGrid, medium: geometry.Medium, omega: float
):
    r"""Returns the standard Helmholtz operator

    ```math
    \left( \nabla^2 + \frac{\omega^2}{c_0^2}\right)\hat p
    ```

    Args:
        grid (geometry.kGrid): [description]
        medium (geometry.Medium): [description]
        omega (float): [description]

    Returns:
        [type]: [description]
    """
    h_laplacian = heterogeneous_laplacian(grid, medium, omega)

    def helmholtz_operator(x, omega, medium):
        laplacian = medium.density * h_laplacian(x, 1.0 / medium.density)
        return laplacian + x * ((omega / medium.sound_speed) ** 2)

    return helmholtz_operator


def get_helmholtz_operator_attenuation(
    grid: geometry.kGrid, medium: geometry.Medium, omega: float
):
    r"""Returns the standard Helmholtz operator

    ```math
    \left( \nabla^2 + \frac{\omega^2}{c_0^2}\right)\hat p
    ```

    Args:
        grid (geometry.kGrid): [description]
        medium (geometry.Medium): [description]
        omega (float): [description]

    Returns:
        [type]: [description]
    """
    laplacian = laplacian_with_pml(grid, medium, omega)

    def helmholtz_operator(x, omega, medium):
        return laplacian(x) + x * (
            ((1 + 1j * medium.attenuation) * omega / medium.sound_speed) ** 2
        )

    return helmholtz_operator


def get_helmholtz_operator_general(
    grid: geometry.kGrid, medium: geometry.Medium, omega: float
):
    r"""Returns the standard Helmholtz operator

    ```math
    \left( \nabla^2 + \frac{\omega^2}{c_0^2}\right)\hat p
    ```

    Args:
        grid (geometry.kGrid): [description]
        medium (geometry.Medium): [description]
        omega (float): [description]

    Returns:
        [type]: [description]
    """
    h_laplacian = heterogeneous_laplacian(grid, medium, omega)

    def helmholtz_operator(x, omega, medium):
        return medium.density*h_laplacian(x, 1.0 / medium.density) + x * (
            ((1 + 1j * medium.attenuation) * (omega / medium.sound_speed)) ** 2
        )

    return helmholtz_operator


def solve_helmholtz(
    grid: geometry.kGrid,
    medium: geometry.Medium,
    src: jnp.ndarray,
    omega: float,
    guess=None,
    method="bicgstab",
    restart=10,
    tol=1e-5,
    solve_method="batched",
    maxiter=None,
    *args,
    **kwgs
) -> jnp.ndarray:
    r"""Solves the Helmholtz equation

    ```math
    \left[\rho\nabla\cdot\left(\frac{1}{\rho}\nabla\right) +
    \left(\frac{\omega(1 + i\tau)}{c}\right)^2\right]u = s
    ```

    where  $`c`$ is the speed of sound map, $`\rho`$ is the density defined in
    `medium.density` and $`\tau`$ is defined in
    `medium.attenuation`.

    Args:
        grid (geometry.kGrid):
        medium (geometry.Medium):
        src (jnp.array): Source field $`s`$, must be complex
        omega (float): Pulsation, as $`\omega = 2\pi f_0`$
        guess ([type], optional): Initial guess for the wavefield $`u`$, must be
            complex and with the same shape as `src`.
        method (str, optional): Solver to use, either `'gmres'` or `'bicgstab'`.
        restart (int, optional): (only for GMRES) If using a Krylov subspace solver, this is
            indicates how many time the Krylov subspace is computed.
        tol ([type], optional): Tollerance for iterative solvers.
        maxiter ([type], optional): maximum number of iterations to be performed.

    Returns:
        The complex wavefield that solves the Helmholtz equation

    !!! example
        ```python
        N = (512, 512); dx = (0.5, 0.5);
        grid = kGrid.make_grid(N, dx)
        medium = Medium(jnp.ones(N), 1., 0., 15)
        src_field = jnp.zeros(N).astype(jnp.complex64)
        src_field = src_field.at[32, 32].set(10.0)
        field = solve_helmholtz(grid, medium, src_field, omega)
        ```
    """
    # Choosing the cheapest Helmholtz operator for the given problem
    if medium.density is None and medium.attenuation is None:
        helmholtz_operator = get_helmholtz_operator(grid, medium, omega)
        src = -1j * omega * src
    elif medium.density is None:  # Heterogeneous attenuation
        helmholtz_operator = get_helmholtz_operator_attenuation(grid, medium, omega)
        src = ((omega ** 2) * medium.attenuation - 1j * omega) * src
    elif medium.attenuation is None:  # General case
        helmholtz_operator = get_helmholtz_operator_density(grid, medium, omega)
        src = -1j * omega * src
    else:
        helmholtz_operator = get_helmholtz_operator_general(grid, medium, omega)
        src = ((omega ** 2) * medium.attenuation - 1j * omega) * src

    # Fixing parameters
    linear_op = partial(helmholtz_operator, omega=omega, medium=medium)

    # Default guess
    if guess is None:
        guess = jnp.zeros(shape=grid.N, dtype=jnp.complex64)  # TODO: proper zeros init

    # Iterative solver
    if method == "gmres":
        field, _ = gmres(
            linear_op,
            src,  # jnp.reshape(src, (-1,)),
            x0=guess,  # jnp.reshape(guess, (-1,)),
            tol=tol,
            solve_method=solve_method,
            restart=restart,
            maxiter=maxiter,
            **kwgs,
        )
    elif method == "bicgstab":
        field, _ = bicgstab(
            linear_op,
            src,  # jnp.reshape(src, (-1,)),
            x0=guess,  # jnp.reshape(guess, (-1,)),
            tol=tol,
            maxiter=maxiter,
            **kwgs,
        )

    return field

def velocity_update_fun(sample_input, grid):
    axes = nparange(-len(grid.N), 0, 1)
    deriv = spectral.derivative_with_k_op(sample_input, grid, -1, axes)

    def d_velocity_dt(rho, c_sq, rho_0):
        p = c_sq * jnp.sum(rho, 0)
        dp = deriv(p)
        return -dp / rho_0
    return d_velocity_dt

def density_update_fun(sample_input, grid):
    axes = nparange(-len(grid.N), 0, 1)
    deriv_funcs = [spectral.derivative_with_k_op(sample_input, grid, 1, ax) for ax in axes]

    def d_density_dt(u, rho_0):
        diag_grad_u = jnp.stack([f(u[ax]) for f,ax in zip(deriv_funcs,axes)])
        return - rho_0*diag_grad_u
    return d_density_dt


def simulate_wave_propagation(
    grid: geometry.kGrid,
    medium: geometry.Medium,
    time_array: geometry.TimeAxis,
    sources: geometry.Sources,
    sensors=None,
    method="spectral",
    output_t_axis=None,
    backprop=False,
    guess=None,
    checkpoint=False
):
    """Simulates wave propagation

    Args:
        grid (geometry.kGrid): [description]
        medium (geometry.Medium): [description]
        time_array (geometry.TimeAxis): [description]
        sources (geometry.Sources): [description]
        sensors ([type], optional): [description]. Defaults to None.
        method (str, optional): [description]. Defaults to "spectral".
        output_t_axis ([type], optional): [description]. Defaults to None.
        backprop (bool, optional): If true, the `vjp` operator can be evaluated, but requires
            a much larger memory footprint

    Returns:
        [type]: [description]
    """
    # Adds the k-space operator to the derivative filters
    grid = grid.to_staggered()
    grid = grid.apply_kspace_operator(jnp.amin(medium.sound_speed), time_array.dt)

    # Making functions for ODE solver
    c_sq = medium.sound_speed ** 2
    dt = time_array.dt

    # Get PML
    pml_alpha = get_time_domain_pml(
        medium.pml_size,
        alpha_max=4,  # 5 / (min(grid.dx) / jnp.amin(medium.sound_speed)),
        dimensions=grid.N,
    )

    # Make decay correction factor
    decay_fact = jnp.exp(-pml_alpha * dt / 2)

    # Represents sensors as a measurement operator on the whole field
    measurement_operator = senstor_to_operator(sensors)

    # Scale mass sources
    c_sq_at_sources = jnp.expand_dims(c_sq[sources.positions], axis=-1)
    source_signals = (
        sources.signals * 2 / (c_sq_at_sources * grid.dx[0])
    )  # TODO: different scalings for non-isotropic grids

    # Get steps to be saved
    if output_t_axis is None:
        output_t_axis = time_array
        t = jnp.arange(0, output_t_axis.t_end + output_t_axis.dt, output_t_axis.dt)
    else:
        t = jnp.arange(0, output_t_axis.t_end + output_t_axis.dt, output_t_axis.dt)
    output_steps = (t / dt).astype(jnp.int32)

    # Integrate wave equation
    N = grid.N
    if guess is None:
        u0 = jnp.zeros((len(N), *N))
        rho0 = jnp.zeros((len(N), *N))
    else:
        u0 = guess[0]
        rho0 = guess[1]

    params = {"rho_0": medium.density, "c_sq": c_sq}

    def get_src_map(idx):
        src = jnp.zeros(N)
        signals = source_signals[:, idx] / len(N)
        src = src.at[sources.positions].add(signals)
        return src
    
    sample_input =get_src_map(0)
    d_velocity_dt = velocity_update_fun(sample_input, grid)
    d_density_dt = density_update_fun(sample_input, grid)

    def du_dt(params, rho, t):
        rho_0 = params["rho_0"]
        c_sq = params["c_sq"]
        return d_velocity_dt(rho, c_sq, rho_0)

    def drho_dt(params, u, t):
        # Make source term
        rho_0 = params["rho_0"]
        rho_update = d_density_dt(u, rho_0)

        idx = (t / dt).round().astype(jnp.int32)
        src = get_src_map(idx)

        return rho_update + src 

    # Checkpoint functions to save memory if requested
    fields = ode.generalized_semi_implicit_euler(
        params,
        du_dt,
        drho_dt,
        measurement_operator,
        decay_fact,
        u0,
        rho0,
        dt,
        output_steps,
        backprop,
        checkpoint,
    )
    return fields


def senstor_to_operator(sensors):
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
