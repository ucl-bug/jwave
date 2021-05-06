from jwave import geometry, signal_processing, spectral, ode
from jwave.geometry import Staggered
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
    sample_input: jnp.ndarray,
    grid: geometry.kGrid,
    medium: geometry.Medium,
    omega: float,
    sigma_max: float = 2.0,
) -> Tuple[Callable, geometry.kGrid]:
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

    # Building derivative operators
    axis = list(range(len(grid.N)))
    D_ops = []
    for ax in axis:
        D, grid = spectral.derivative_init(sample_input, grid, staggered=Staggered.NONE, axis=ax)
        
        D_ops.append(D)

    degrees = jnp.array([1.0, 2.0])

    # Constructing laplacian operator. Note that the
    def lapl(field, grid):
        res = jnp.zeros_like(field)
        for ax in axis:
            # Returns the first and second order derivatives
            derivatives = jax.vmap(D_ops[ax], in_axes=(None, None, 0))(
                field, grid, degrees
            )
            D_ax = derivatives[0]
            D2_ax = derivatives[1]
            lapl_term = D2_ax / (gamma[ax] ** 2) - gamma_prime[ax] * D_ax / (
                gamma[ax] ** 3
            )
            res = res + lapl_term
        return res

    return lapl, grid


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
    # TODO: This uses meshgrid to duplicate the PML function across the various dimensions.
    #       it would be better to use broadcasting to reduce memory requirements.

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


def heterogeneous_laplacian(
    sample_input: jnp.ndarray,
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

    # Building derivative operators
    axis = list(range(sample_input.ndim))
    D_ops = []
    for ax in axis:
        D, grid = spectral.derivative_init(sample_input, grid, staggered=0, axis=ax)
        D_ops.append(D)

    axis = list(range(sample_input.ndim))

    def lapl(field, cmap, grid):
        # TODO: This for loop probably forces sequential computations. However, it reduces
        #       the memory requirements. Better parallelization tests are needed
        res = jnp.zeros_like(field)
        for ax in axis:
            res = res + D_ops[ax](cmap * (D_ops[ax](field, grid, 1.) / gamma[ax]), grid, 1.) / gamma[ax]
        return res

    return lapl, grid


def get_helmholtz_operator(
    sample_input: jnp.ndarray,
    grid: geometry.kGrid,
    medium: geometry.Medium,
    omega: float,
):
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
    laplacian, grid = laplacian_with_pml(sample_input, grid, medium, omega)
    

    def helmholtz_operator(x, omega, medium, grid):
        return laplacian(x, grid) + x * ((omega / medium.sound_speed) ** 2)
    
    return helmholtz_operator, grid


def get_helmholtz_operator_density(
    sample_input: jnp.ndarray,
    grid: geometry.kGrid,
    medium: geometry.Medium,
    omega: float,
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
    h_laplacian, grid = heterogeneous_laplacian(sample_input, grid, medium, omega)

    def helmholtz_operator(x, omega, medium, grid):
        laplacian = medium.density * h_laplacian(x, 1.0 / medium.density, grid)
        return laplacian + x * ((omega / medium.sound_speed) ** 2)

    return helmholtz_operator, grid


def get_helmholtz_operator_attenuation(
    sample_input: jnp.ndarray,
    grid: geometry.kGrid,
    medium: geometry.Medium,
    omega: float,
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
    laplacian, grid = laplacian_with_pml(sample_input, grid, medium, omega)

    def helmholtz_operator(x, omega, medium, grid):
        return laplacian(x, grid) + x * (
            ((1 + 1j * medium.attenuation) * omega / medium.sound_speed) ** 2
        )

    return helmholtz_operator, grid


def get_helmholtz_operator_general(
    sample_input: jnp.ndarray,
    grid: geometry.kGrid,
    medium: geometry.Medium,
    omega: float,
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
    h_laplacian, grid = heterogeneous_laplacian(sample_input, grid, medium, omega)

    def helmholtz_operator(x, omega, medium, grid):
        return medium.density * h_laplacian(x, 1.0 / medium.density, grid) + x * (
            ((1 + 1j * medium.attenuation) * (omega / medium.sound_speed)) ** 2
        )

    return helmholtz_operator, grid

def solve_helmholtz(
    grid: geometry.kGrid,
    medium: geometry.Medium,
    src: jnp.ndarray,
    omega: float,
    guess=None,
    method="gmres",
    restart=10,
    tol=1e-5,
    solve_method="batched",
    maxiter=None,
    solve_problem=True,
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

    # Default guess
    if guess is None:
        guess = jnp.zeros(shape=grid.N, dtype=jnp.complex64)  # TODO: proper zeros init

    # Operators and scaling
    if medium.density is None and medium.attenuation is None:
        helmholtz_operator, grid = get_helmholtz_operator(guess, grid, medium, omega)
        

        def scale_src(src, omega):
            return -1j * omega * src

    elif medium.density is None:  # Heterogeneous attenuation
        helmholtz_operator, grid = get_helmholtz_operator_attenuation(
            guess, grid, medium, omega
        )

        def scale_src(src, omega):
            return ((omega ** 2) * medium.attenuation - 1j * omega) * src

    elif medium.attenuation is None:  # General case
        helmholtz_operator, grid = get_helmholtz_operator_density(guess, grid, medium, omega)

        def scale_src(src, omega):
            return -1j * omega * src

    else:
        helmholtz_operator, grid = get_helmholtz_operator_general(guess, grid, medium, omega)

        def scale_src(src, omega):
            return ((omega ** 2) * medium.attenuation - 1j * omega) * src

    # Iterative solver
    params = {
        "grid": grid,
        "src": src,
        "guess": guess,
        "medium": medium,
        "omega": omega,
        "solver_params": {
            "tol": tol,
            "restart": restart,
            "solve_method": solve_method,
            "restart": restart,
            "maxiter": maxiter,
        },
    }

    # Returns operator and parameters if method is None
    if method is None:
        return params, [helmholtz_operator, scale_src]

    # Construct solver
    if method == "gmres":

        def solver(params):
            linear_op = partial(
                helmholtz_operator, omega=params["omega"], medium=params["medium"], grid=params["grid"]
            )
            src = scale_src(params["src"], params["omega"])
            return gmres(
                linear_op,
                src,  # jnp.reshape(src, (-1,)),
                x0=params["guess"],  # jnp.reshape(guess, (-1,)),
                tol=params["solver_params"]["tol"],
                solve_method=params["solver_params"]["solve_method"],
                restart=params["solver_params"]["restart"],
                maxiter=params["solver_params"]["maxiter"],
            )[0]

    elif method == "bicgstab":

        def solver(params):
            linear_op = partial(helmholtz_operator, omega=params["omega"], medium=params["medium"], grid=params["grid"])
            src = scale_src(params["src"], params["omega"])
            return bicgstab(
                linear_op,
                src,  # jnp.reshape(src, (-1,)),
                x0=params["guess"],  # jnp.reshape(guess, (-1,)),
                tol=params["solver_params"]["tol"],
                maxiter=params["solver_params"]["maxiter"],
            )[0]

    if solve_problem:
        return solver(params)
    else:
        return params, solver


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
    deriv_funcs = [
        spectral.derivative_with_k_op(sample_input, grid, 1, ax) for ax in axes
    ]

    def d_density_dt(u, rho_0):
        diag_grad_u = jnp.stack([f(u[ax]) for f, ax in zip(deriv_funcs, axes)])
        return -rho_0 * diag_grad_u

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
    checkpoint=False,
    get_function=False,
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
        u0 = jnp.zeros((len(N), *N), dtype=c_sq.dtype)
        rho0 = jnp.zeros((len(N), *N), dtype=c_sq.dtype)
    else:
        u0 = guess[0]
        rho0 = guess[1]

    acoustic_params = {"rho_0": medium.density, "c_sq": c_sq}

    sample_input = jnp.zeros_like(c_sq)
    d_velocity_dt = velocity_update_fun(sample_input, grid)
    d_density_dt = density_update_fun(sample_input, grid)

    # Parameters dictionary
    params = {
        "acoustic_params": acoustic_params,
        "source_signals": source_signals,
        "initial_wavefields": {"u0": u0, "rho0": rho0},
    }

    # Building solver function
    def solver_function(params):
        acoustic_params = params["acoustic_params"]
        source_signals = params["source_signals"]
        u0 = params["initial_wavefields"]["u0"]
        rho0 = params["initial_wavefields"]["rho0"]

        # Function to integrate
        def get_src_map(idx):
            src = jnp.zeros(N)
            signals = source_signals[:, idx] / len(N)
            src = src.at[sources.positions].add(signals)
            return src

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

        # Integrate
        fields = ode.generalized_semi_implicit_euler(
            acoustic_params,
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

    if get_function:
        return solver_function, params
    else:
        return solver_function(params)


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
