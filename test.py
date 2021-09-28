from jaxdf import operators as jops
from jaxdf.core import operator, Field
from jaxdf.discretization import FourierSeries, Coordinate
from jaxdf.geometry import Domain
from jaxdf.utils import join_dicts
from jax import numpy as jnp
from jax.scipy.sparse.linalg import gmres
from jax.experimental import optimizers
import jax

# Settings
domain = Domain((256, 256), (1., 1.))
seed = jax.random.PRNGKey(0)

# Speed of sound parametrization
lens_params = jax.random.uniform(seed, (168,40))
def get_sos(p):
    lens = jnp.zeros(domain.N).at[44:212,108:148].set(jax.nn.sigmoid(p)) + 1
    return jnp.expand_dims(lens, -1)

# Defining operators
@jops.elementwise
def pml_absorption(x):
    abs_x = jnp.abs(x)
    return jnp.where(abs_x > 110, (jnp.abs(abs_x-110)/(128. - 110)), 0.)**2

gamma = lambda x: 1./(1 + 1j*pml_absorption(x))

@operator()
def helmholtz(u, c, x):
    pml = gamma(x)
    mod_grad_u = jops.gradient(u)*pml
    mod_diag_jacobian = jops.diag_jacobian(mod_grad_u)*pml
    laplacian = jops.sum_over_dims(mod_diag_jacobian)
    return laplacian + ((1./c)**2)*u

@operator()
def integrand_TV(u):
    nabla_u = jops.gradient(u)    
    return jops.sum_over_dims(jops.elementwise(jnp.abs)(nabla_u))

# Defining discretizations
fourier_discr = FourierSeries(domain)
u_fourier_params, u = fourier_discr.empty_field(name='u')
src_fourier_params, src = fourier_discr.empty_field(name='src')
src_fourier_params = u_fourier_params.at[128, 40].set(1. + 0j)  # Monopole source
_, c = fourier_discr.empty_field(name='c')
_, x = fourier_discr.empty_field(name='x')
x_params = Coordinate(domain).get_field_on_grid()({}) # Coordinate field

# Discretizing operators: getting pure functions and parameters
H = helmholtz(u=u, c=c, x=x)
TV = integrand_TV(u=u)
global_params = join_dicts(H.get_global_params(), TV.get_global_params())
H_on_grid = H.get_field_on_grid(0)
tv_on_grid = lambda x: TV.get_field_on_grid(0)(global_params, x)

# Helmholtz solver function
def solve_helmholtz(speed_of_sound):
    params = {"c":speed_of_sound, "x":x_params}
    def helm_func(u):
        params["u"] = u
        return H_on_grid(global_params, params)
    sol, _ = gmres(helm_func, src_fourier_params, maxiter=1000, tol=1e-3, restart=50)
    return sol

# Loss function
def loss(p):
    sos = get_sos(p)
    tv_term = jnp.mean(H_on_grid(sos))
    field = solve_helmholtz(sos)
    return -jnp.sum(jnp.abs(field[70,210])) + 1e-4*tv_term

# Optimization loop
init_fun, update_fun, get_params = optimizers.adam(.1, b1=0.9, b2=0.9)
opt_state = init_fun(lens_params)

@jax.jit
def update(opt_state, k):
    lossval, gradient = jax.value_and_grad(loss)(get_params(opt_state))
    return lossval, update_fun(k, gradient, opt_state)

for k in range(100):
    lossval, opt_state = update(opt_state, k)