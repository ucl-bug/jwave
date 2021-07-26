from jwave.discretization import RealFourierSeries
from jax import random
from jwave.geometry import Domain
import jax
from jwave.core import Field
from jwave.core import operator

seed = random.PRNGKey(42)

domain = Domain((1024,1024), (.5,.6))
fourier_discretization = RealFourierSeries(domain)

@operator(debug=False)
def custom_op(u):
    return u + 2

# Fourier discretization
seeds = random.split(seed, 2)
u_params = fourier_discretization.random_field(seeds[0])
u = Field(fourier_discretization, params=u_params, name='u')

# Compiling operator on the given discretization
op = custom_op(u=u)
print(op)