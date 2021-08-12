from jax import random
from jwave.geometry import Domain
from jax import numpy as jnp
from jwave.core import operator, Field
from jwave.discretization import Arbitrary, RealFourierSeries
from jwave import operators as jops
from jax.experimental import stax

# Hyperparameters
seed = random.PRNGKey(42)
seeds = random.split(seed, 10)
domain = Domain((32, 35), (0.5, 0.6))
x = jnp.array([1.0, 2.0])

# Fields
init_random_params, predict = stax.serial(
    stax.Dense(1024), stax.Relu, stax.Dense(1024), stax.Relu, stax.Dense(1)
)


def init_params(seed, domain):
    return init_random_params(seed, (len(domain.N),))[1]


def get_fun(params, x):
    return predict(params, x)


arbitrary_discr = Arbitrary(domain, get_fun, init_params)
arbitrary_field_u, u_arbitrary = arbitrary_discr.random_field(seeds[0], 'u')
arbitrary_field_v, v_arbitrary = arbitrary_discr.random_field(seeds[1], 'v')

fourier_discr = RealFourierSeries(domain)
fourier_field_u, u_fourier = fourier_discr.random_field(seeds[0], 'u')
fourier_field_v, v_fourier = fourier_discr.random_field(seeds[1], 'v')


def _apply_operator(op):
    out_field = op(u=u_arbitrary)
    global_params = out_field.get_global_params()
    out_field.get_field(0)(global_params, {"u": arbitrary_field_u}, x)

    out_field = op(u=u_fourier)
    global_params = out_field.get_global_params()
    _ = out_field.get_field(0)(global_params, {"u": fourier_field_u}, x)


def _apply_binary_operator(op):
    out_field = op(u=u_arbitrary, v=v_arbitrary)
    global_params = out_field.get_global_params()
    out_field.get_field(0)(
        global_params, {"u": arbitrary_field_u, "v": arbitrary_field_v}, x
    )

    out_field = op(u=u_fourier, v=v_fourier)
    global_params = out_field.get_global_params()
    _ = out_field.get_field(0)(
        global_params, {"u": fourier_field_u, "v": fourier_field_v}, x
    )


def test_call():
    """This can't be jitted"""
    u_arbitrary(x)
    u_fourier(x)


def test_get_field():
    u_arbitrary.get_field()(arbitrary_field_u, x)
    u_fourier.get_field()(fourier_field_u, x)


def test_add_scalar():
    @operator()
    def op(u):
        return u + 1.0

    _apply_operator(op)


def test_several():
    @operator()
    def op(u):
        Tanh = jops.elementwise(jnp.tanh)
        return Tanh(u) + u + 2 * u

    _apply_operator(op)


def test_product_and_sum():
    @operator()
    def op(u):
        return u * u + u

    _apply_operator(op)


def test_two_inputs():
    @operator()
    def op(u, v):
        return u * v + u

    _apply_binary_operator(op)


# This is for debugging
if __name__ == "__main__":
    test_call()
    test_two_inputs()
