from timeit import repeat

import jax
from diffrax import (ConstantStepSize, ODETerm, RecursiveCheckpointAdjoint,
                     SaveAt, SemiImplicitEuler, diffeqsolve)
from jax import numpy as jnp
from jax.lax import scan


def _speedtest(integrator, grad_fn, fields, t, name):
    integration_compiled = jax.jit(integrator).lower(fields, t).compile()
    integration_times = repeat(
        lambda: integration_compiled(fields, t)[0].block_until_ready(),
        number=5,
        repeat=20)
    print(f"{name}: {min(integration_times)}")

    grad_fn_compiled = jax.jit(grad_fn).lower(fields, t).compile()
    scan_times = repeat(
        lambda: grad_fn_compiled(fields, t)[0].block_until_ready(),
        number=5,
        repeat=20)
    print(f"{name} AD: {min(scan_times)}")


##### SETUP #####
N = 256
N_steps = 2000
t = jnp.linspace(0, 1, N_steps)

u0, v0 = jnp.zeros((N, N)), jnp.zeros((N, N)).at[32, 32].set(1.0)
fields = (u0, v0)

# Integration terms
du = lambda t, v, args: -(v**2)
dv = lambda t, u, args: -jnp.fft.irfft(jnp.sin(jnp.fft.rfft(u)))
sample = lambda t, y, args: y[0][64, 64]    # Some arbitrary sampling function


##### INTEGRATE WITH scan #####
@jax.checkpoint
def scan_fun(carry, t):
    u, v, dt = carry
    u = u + du(t, v, None) * dt
    v = v + dv(t, u, None) * dt
    return (u, v, dt), sample(t, (u, v), None)


def integrator(fields, t):
    dt = t[1] - t[0]
    carry = (fields[0], fields[1], dt)
    _, values = scan(scan_fun, carry, t)
    return values


@jax.grad
def grad_fn(fields, t):
    return jnp.mean(integrator(fields, t)**2)


# Timing
_speedtest(integrator, grad_fn, fields, t, "scan")

##### INTEGRATE WITH SemiImplicitEuler #####
terms = ODETerm(du), ODETerm(dv)


def integrator(fields, t):
    return diffeqsolve(terms,
                       solver=SemiImplicitEuler(),
                       t0=t[0],
                       t1=t[-1],
                       dt0=t[1] - t[0],
                       y0=fields,
                       args=None,
                       saveat=SaveAt(steps=True, fn=sample, dense=False),
                       stepsize_controller=ConstantStepSize(),
                       adjoint=RecursiveCheckpointAdjoint(checkpoints=N_steps),
                       max_steps=N_steps).ys


@jax.grad
def grad_fn(fields, t):
    return jnp.mean(integrator(fields, t)**2)


# Timing
_speedtest(integrator, grad_fn, fields, t, "SemiImplicitEuler")
