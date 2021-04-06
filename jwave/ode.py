import jax.numpy as jnp
from jax import jit
from functools import partial
import jax
from typing import Callable


def _identity(x):
    return x


def euler_integration(f, x0, dt, output_steps, timestep_correction=1):
    r"""Integrates the differential equation

    ```math
    \dot x = f(x,t)
    ```
    using a [first-order Euler method](https://en.wikipedia.org/wiki/Euler_method).
    The solution $`x`$ is given by

    ```math
    x^{(n+1)} = x^{(n)} + f\left(x^{(n)}, t^{(n)}\right)dt\cdot\kappa
    ```

    The structure of $`x`$ is inferred by $`x^{(0)}`$, which can be any pytree.
    The output of $`f`$ should be compatible with $`x^{(0)}`$ (must have the same
    [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) structure)

    `output_steps` must be the index of the steps to save. $`\kappa`$ is `timestep_correction`.

    The stepsize of the euler integration is given by $`\kappa\cdot dt`$.

    Args:
        f (function): Differential equation to integrate
        x0 (pytree): Initial point
        dt (float): Time-step
        output_steps ([int]): Iterations of the euler solver to save
        timestep_correction (int, optional): This scales the dt by a given factor
            when applying the euler step. Defaults to 1.

    Returns:
        [pytree]: List of pytree, one for each value of `output_steps`. The
            structure of each pytree is the same as $`x^{(0)}`$.

    !!! example
        ```python
        from jwave.ode import euler_integration

        # Simulation settings
        g = 9.81
        dt = 0.01
        output_steps = jnp.array([0,5])/dt # Must be in steps, not seconds

        # Newton law of motion
        f = lambda (x, v), t: v, 0.5*g

        # Initial conditions
        x0 = 1.
        v0 = 0.

        # Calculate endpoint
        x_end, v_end = euler_integration(f, (x0,v0), dt, output_steps)
        ```

    """
    # TODO: make output_steps in seconds (and interpolate?)

    # Create vectors of (positive) indices to return
    assert any(map(lambda x: x >= 0, output_steps))
    return _euler_integration(f, x0, dt, output_steps, timestep_correction)


@partial(jit, static_argnums=(0,))
def _euler_integration(f, x0, dt, output_steps, timestep_correction):
    # Step correction
    k = timestep_correction

    def euler_step(i, x):
        dx_dt = f(x, i * dt)
        return jax.tree_util.tree_multimap(
            lambda x, y: jnp.add(x, y * dt * k), x, dx_dt
        )

    def euler_jump(x_t, i):
        x = x_t[0]
        start = x_t[1]
        end = start + i

        y = jax.lax.fori_loop(start, end, euler_step, x)
        return (y, end), y

    jumps = jnp.diff(output_steps)

    _, ys = jax.lax.scan(euler_jump, (x0, 0.0), jumps)
    return ys


def semi_implicit_euler(f, g, x0, y0, dt, output_steps, timestep_correction=1):
    r"""Integrates a system of differential equations having the form
  ```math
  \begin{dcases}
      \dot x = f(y,t) \\
      \dot y = g(x,t)
  \end{dcases}
  ```
  The solution is computed using a first-order semi-implicit Euler
  integrator, which almost conserves the energy for time-independent
  equations, as follows
  ```math
  \begin{dcases}
      x^{(n+1)} &= x^{(n)}   &+& f\left(y^{(n)}, t^{(n)}\right)dt\cdot \kappa \\
      y^{(n+1)} &= y^{(n)} &+& g\left(x^{(n+1)}, t^{(n)}\right)dt\cdot \kappa
  \end{dcases}
  ```

  The structure of $`x`$ is inferred by $`x^{(0)}`$, which can be any pytree.
  The output of $`f`$ should be compatible with $`x^{(0)}`$ (must have the same 
  [pytree](https://jax.readthedocs.io/en/latest/pytrees.html) structure). A 
  symmetric argument holds for $`g`$.

  `output_steps` must be the index of the steps to save. $`\kappa`$ is `timestep_correction`.

  The stepsize of the euler integration is given by $`\kappa\cdot dt`$.

  Args:
      f (function): First ODE to integrate
      g (function): Second ODE to integrate
      x0 (pytree): First initial value
      y0 (pytree): Second initial value
      dt (float): Stepsize in seconds
      output_steps ([int]): Iterations of the euler solver to save
      timestep_correction (int, optional): This scales the dt by a given factor 
          when applying the euler step. Defaults to 1.

  Returns:
      [[pytree],[pytree]]: Pair of lists of pytree, representing the two conjugate
          variables $`(x,y)`$. For each one, the function returns a list with one
          element for each value of `output_steps`. The structure of each pytree 
          is the same as $`x^{(0)}`$.
  """

    # Create vectors of (positive) indices to return
    assert any(map(lambda x: x >= 0, output_steps))
    return _semi_implicit_euler(f, g, x0, y0, dt, output_steps, timestep_correction)


@partial(jit, static_argnums=(0, 1))
def _semi_implicit_euler(f, g, x0, y0, dt, output_steps, timestep_correction):
    # Step correction
    k = timestep_correction

    def euler_step(i, conj_variables):
        x, y = conj_variables
        dx_dt = f(y, i * dt)
        x = jax.tree_util.tree_multimap(lambda x, y: jnp.add(x, y * dt * k), x, dx_dt)
        dy_dt = g(x, i * dt)
        y = jax.tree_util.tree_multimap(lambda x, y: jnp.add(x, y * dt * k), y, dy_dt)
        return (x, y)

    def euler_jump(x_t, i):
        x = x_t[0]
        start = x_t[1]
        end = start + i

        y = jax.lax.fori_loop(start, end, euler_step, x)
        return (y, end), y

    jumps = jnp.diff(output_steps)

    _, ys = jax.lax.scan(euler_jump, ((x0, y0), 0.0), jumps)
    return ys


def generalized_semi_implicit_euler(
    params,
    f: Callable,
    g: Callable,
    measurement_operator: Callable,
    alpha: jnp.ndarray,
    x0: jnp.ndarray,
    y0: jnp.ndarray,
    dt: float,
    output_steps: jnp.ndarray,
    backprop=False,
    checkpoint=False
):
    r"""This functions works in the same way as the
    [`semi_implicit_euler`](#jwave.ode.semi_implicit_euler) integrator,
    with the difference that the update function accepts an extra
    parameter $`\alpha`$ with the same pytree-structure as $`x`$ and 
    $`y`$. 

    Variable update is performed as 
    ```math
    \begin{dcases}
        x^{(n+1)} &= \alpha\left[\alpha x^{(n)}   + f\left(y^{(n)}, t^{(n)}\right)dt\cdot\right] \\
        y^{(n+1)} &= \alpha\left[\alpha y^{(n)} + g\left(x^{(n+1)}, t^{(n)}\right)dt\cdot\right] \\
        r^{(n)} &= M(x^{(n+1)}, y^{(n+1)})
    \end{dcases}
    ```

    $`M(x,y)`$ is an arbitrary measurement operator that is applied at
    the end of each timestep, for example it could evaluate the pressure intensity
    or the field value at some specific locations. If `None`, defaults to the identity 
    operator $`I(x,y)=(x,y)`$. 
    The vector of measurements $`r`$=`r` is returned. 

    !!! warning
        Calling this method with `backprop=True` allows to perform backpropagation.
        However, this requires storing the entire wavefield history and is therefore
        memory demanding. Alternatively, `backprop=False` allows to calculate
        derivatives using forward-propagation, or jacobian-vector products
        with memory cost independent of the simulation length. 
        Combined with `jax.vmap` or `jax.jaxfwd`, this makes easy to calculate
        gradients for functions that have tall jacobians, such as simulations 
        that depends on a small amount of parameters (e.g. delays, steering angle, 
        etc)

    Args:
        f (Callable): 
        g (Callable): 
        alpha (jnp.ndarray): 
        x0 (jnp.ndarray): 
        y0 (jnp.ndarray): 
        dt (float): [description]
        output_steps (jnp.ndarray): 
        measurement_operator ([type], optional): Defaults to `None`
        backprop (bool, optional): If true, the `vjp` operator can be evaluated, but requires
            a much larger memory footprint (all forward fields must be stored)

    !!! example
        ```python
        # Integrating the equations of motions for a planet around a star
        M_sun = 2.0  # kg
        p0 = jnp.array([0.0, 3.0])  # m
        v0 = jnp.array([1.0, 0.0])  # m/s
        G = 1
        dt = 0.1
        t_end = 200.0
        output_steps = (jnp.arange(0, t_end, 10 * dt) / dt).round()

        # Equations of motion
        f_1 = lambda v, t: v
        f_2 = lambda p, t: newton_grav_law(G=1, M=M_sun, r=p)

        M = lambda x: x # Identity operator, could have been `None`

        # Integrate
        trajectory , _ = generalized_semi_implicit_euler(
            f = f_1, 
            g = f_2, 
            measurement_operator = M, 
            alpha=0.0,
            x0 = p0,
            y0 = v0,
            dt = dt,
            output_steps = output_steps,
            backprop=False
        )
        ```
    """

    # Create vectors of (positive) indices to return
    # assert any(map(lambda x: x >= 0, output_steps))
    if measurement_operator is None:
        measurement_operator = _identity

    if backprop:
        return _generalized_semi_implicit_euler_with_vjp(
            params, f, g, measurement_operator, alpha, x0, y0, dt, output_steps, checkpoint
        )
    else:
        return _generalized_semi_implicit_euler(
            params, f, g, measurement_operator, alpha, x0, y0, dt, output_steps
        )


def variable_update_with_pml(x, dx_dt, k, dt):
    x = jax.tree_util.tree_multimap(lambda x, y, a: a * (a * x + y * dt), x, dx_dt, k)
    return x


@partial(jit, static_argnums=(1, 2, 3))
def _generalized_semi_implicit_euler(
    params, f, g, measurement_operator, k, x0, y0, dt, output_steps
):
    def euler_step(i, conj_variables):
        x, y = conj_variables
        dx_dt = f(params, y, i * dt)
        x = variable_update_with_pml(x, dx_dt, k, dt)
        dy_dt = g(params, x, i * dt)
        y = variable_update_with_pml(y, dy_dt, k, dt)
        return (x, y)

    def euler_jump(x_t, i):
        x, start = x_t
        end = start + i

        y = jax.lax.fori_loop(start, end, euler_step, x)
        return (y, end), measurement_operator(y)

    jumps = jnp.concatenate([jnp.diff(output_steps), jnp.array([1])])

    _, ys = jax.lax.scan(euler_jump, ((x0, y0), 0.0), jumps)
    return ys


# @partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _generalized_semi_implicit_euler_with_vjp(
    params, f, g, measurement_operator, k, x0, y0, dt, output_steps, checkpoint
):
    def step_without_measurements(carry, t):
        x, y = carry
        dx_dt = f(params, y, t * dt)
        x = variable_update_with_pml(x, dx_dt, k, dt)
        dy_dt = g(params, x, t * dt)
        y = variable_update_with_pml(y, dy_dt, k, dt)
        return (x, y)

    def single_step(carry, t):
        fields = step_without_measurements(carry, t)
        return fields, measurement_operator(fields)

    if checkpoint:
        single_step = jax.checkpoint(single_step)

    _, ys = jax.lax.scan(single_step, (x0, y0), output_steps)

    return ys


"""
def _generalized_semi_implicit_euler_fwd(
    params, f, g, measurement_operator, k, x0, y0, dt, output_steps
):
    def step_without_measurements(carry, t):
        x, y = carry
        dx_dt = f(params, y, t * dt)
        x = variable_update_with_pml(x, dx_dt, k, dt)
        dy_dt = g(params, x, t * dt)
        y = variable_update_with_pml(y, dy_dt, k, dt)

        return (x, y, t * dt)

    def single_step(carry, t):
        x, y, t = step_without_measurements(carry, t)
        return (x, y), (x, y, t, measurement_operator((x, y)))

    _, ys = jax.lax.scan(single_step, (x0, y0), output_steps)
    x_t, y_t, t, measurements = ys

    res = {
        "fields": (x_t, y_t),
        "params": params,
        "k": k,
        "t": t,
        "dt": dt,
        "output_steps": output_steps,
        "shapes": [k.shape, x0.shape, y0.shape, dt.shape, output_steps.shape],
    }
    return measurements, res


def _generalized_semi_implicit_euler_bwd(
    f, g, measurement_operator, res, measurements_bar
):
    measurements_bar = jax.device_put(measurements_bar)
    x_t, y_t = res["fields"]
    x_shape = x_t.shape[1:]
    y_shape = y_t.shape[1:]

    params = res["params"]
    _, m_vjp = jax.vjp(measurement_operator, (jnp.zeros(x_shape), jnp.zeros(y_shape)))

    time_axis = res["t"]
    k = res["k"]
    dt = res["dt"]
    output_steps = res["output_steps"]

    a_x = jnp.zeros(x_shape)
    a_y = jnp.zeros(y_shape)
    a_params = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)

    def reversed_step(adjoints, n):
        a_x, a_y, a_params = adjoints
        m = (measurements_bar[0][n], measurements_bar[1][n])
        dL_dx, dL_dy = m_vjp(m)[0]

        # Getting fields
        x = x_t[n]
        y = y_t[n]
        t_step = time_axis[n]

        # Making VJPs
        _, vjp_f = jax.vjp(f, params, y, t_step)
        _, vjp_g = jax.vjp(g, params, x, t_step)

        # Update adjoint fields with PML and parameters
        a_x = k * (k * a_x - dt * (vjp_f(a_y)[1] + dL_dx))
        a_y = k * (k * a_y - dt * (vjp_g(a_x)[1] + dL_dy))

        # Update adjoint parameters
        delta_params = vjp_f(a_x)[0]
        a_params = jax.tree_util.tree_multimap(
            lambda x, d: x - dt * d, a_params, delta_params
        )
        delta_params = vjp_g(a_y)[0]
        a_params = jax.tree_util.tree_multimap(
            lambda x, d: x - dt * d, a_params, delta_params
        )
        return (a_x, a_y, a_params), None

    carry, _ = jax.lax.scan(
        reversed_step, (a_x, a_y, a_params), output_steps, reverse=True
    )
    a_x, a_y, a_params = carry
    a_params = jax.tree_util.tree_map(lambda x: -x, a_params)

    # All non-param gradients are zero
    empty_stuff = list(map(lambda x: jnp.zeros(x), res["shapes"]))

    duals = (a_params, *empty_stuff)

    return duals


_generalized_semi_implicit_euler_with_vjp.defvjp(
    _generalized_semi_implicit_euler_fwd, _generalized_semi_implicit_euler_bwd
)
"""

if __name__ == "__main__":
    pass
