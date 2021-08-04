from jax import random
from jwave.geometry import Domain
import jax
from jax import numpy as jnp
from jwave.core import operator, Field
from jwave.discretization import Arbitrary, RealFourierSeries
from jwave import operators as jops
from jax import numpy as jnp
from jax.experimental import stax

# Hyperparameters
seed = random.PRNGKey(42)
domain = Domain((32,35), (.5,.6))
x = jnp.array([1., 2.])

# Fields
init_random_params, predict = stax.serial(
    stax.Dense(1024), stax.Relu,
    stax.Dense(1024), stax.Relu,
    stax.Dense(1))
init_params = lambda seed, domain: init_random_params(seed, (len(domain.N),))[1]
def get_fun(params, x):
    return predict(params, x)
arbitrary_discr = Arbitrary(domain, get_fun, init_params)
arbitrary_field = arbitrary_discr.random_field(seed)
u_arbitrary = Field(arbitrary_discr, params=arbitrary_field, name='u')

fourier_discr = RealFourierSeries(domain)
fourier_field = fourier_discr.random_field(seed)
u_fourier = Field(fourier_discr, params=fourier_field, name='u')

def _apply_operator(op):
    out_field = op(u=u_arbitrary)
    global_params = out_field.get_global_params()
    out_field.get_field(0)(global_params, {"u": arbitrary_field}, x)

    out_field = op(u=u_fourier)
    global_params = out_field.get_global_params()
    _ = out_field.get_field(0)(global_params, {"u": fourier_field}, x)

def test_call():
    """This can't be jitted"""
    u_arbitrary(x)
    u_fourier(x)

def test_get_field():
    u_arbitrary.get_field()(arbitrary_field,x)
    u_fourier.get_field()(fourier_field,x)

def test_add_scalar():
    @operator()
    def op(u):
        return u + 1.
    _apply_operator(op)

def test_several():
    @operator()
    def op(u):
        Tanh = jops.elementwise(jnp.tanh)
        return Tanh(u) + u + 2*u 
    _apply_operator(op)


# This is for debugging
if __name__ == '__main__':
    test_several()