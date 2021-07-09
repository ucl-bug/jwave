from jwave.geometry import Domain
from jwave.core import make_op, Discretization, Field
from jwave import spectral
from functools import reduce, wraps
from jax import random, vmap, jvp
from jax import numpy as jnp
from typing import Callable
from sympy import symbols, Function, factorial, ZeroMatrix
import numpy as np

def _build_arbitrary_operator(
    get_fun, 
    parameter_fun, 
    domain, 
    parameter_fun_name='_',
    static_argnums=[]
):
    # Make get function
    def init_params(seed):
        raise NotImplementedError("Please initialize the parameters of the input fields instead.")
    discretization = Arbitrary(domain, get_fun, init_params)

    # Preprocess parameters function
    _fun = make_op(discretization, static_argnums, name=parameter_fun_name)(parameter_fun)
    return _fun

def _constant_param_fun(get_fun, u):
    param_fun = lambda u_p: u_p
    op = _build_arbitrary_operator(
        get_fun, param_fun, u.discretization.domain, parameter_fun_name='identity')
    return op(u)

def _join_parameter_fun(get_fun, u, v):
    param_fun = lambda u_p, v_p: [u_p, v_p]
    op = _build_arbitrary_operator(
        get_fun, param_fun, u.discretization.domain, parameter_fun_name='join')
    return op(u,v)


class Arbitrary(Discretization):
    def __init__(self, domain: Domain, get_fun: Callable, init_params: Callable):
        self._get_fun = get_fun
        self._init_params = init_params
        self.domain = domain
        self.params["coordinate_grid"] = self.domain.grid

    def update_params(self, name, value):
        if name in self.params.keys():
            raise ValueError("Parameter {} already exists".format(name))
        self.params[name] = value

    @staticmethod
    def add(u,v):
        f_u = u.discretization.get_field()
        f_v = v.discretization.get_field()
        def get_fun(dp, up, x):
            return f_u(dp,up[0],x) + f_v(dp,up[1],x)
        return _join_parameter_fun(get_fun, u, v)
    
    @staticmethod    
    def add_scalar(u, number):
        # Make get function
        f_u = u.discretization.get_field()
        get_fun = lambda dp, up, x: f_u(dp,up,x) + number
        return _constant_param_fun(get_fun, u)

    @staticmethod
    def elementwise(u, function):
        f_u = u.discretization.get_field()
        get_fun = lambda dp, up, x: function(f_u(dp,up,x))
        return _constant_param_fun(get_fun, u)

    @staticmethod
    def derivative(u, axis):
        f_u = u.discretization.get_field()
        ndim = u.discretization.domain
        tangents = jnp.zeros((ndim,))
        tangents = tangents.at[axis].set(1.)
        def get_fun(dp, up, x):
            fun = lambda position: f_u(dp,up,position)
            return jvp(fun, x, tangents)
        return _constant_param_fun(get_fun, u)
    
    @staticmethod
    def invert( u):
        f_u = u.discretization.get_field()
        get_fun = lambda dp, up, x: -f_u(dp,up,x)
        param_fun = lambda u_p: u_p
        op = _build_arbitrary_operator(get_fun, param_fun, u.discretization.domain)
        return op(u)
    
    @staticmethod
    def mul(u,v):
        f_u = u.discretization.get_field()
        f_v = v.discretization.get_field()
        get_fun = lambda dp, up, x: f_u(dp,up[0],x)*f_v(dp,up[1],x)
        return _join_parameter_fun(get_fun, u, v)
    
    @staticmethod
    def mul_scalar(u, number):
        f_u = u.discretization.get_field()
        get_fun = lambda dp, up, x: f_u(dp,up,x)*number
        return _constant_param_fun(get_fun, u)
    
    @staticmethod
    def power( u, v):
        f_u = u.discretization.get_field()
        f_v = v.discretization.get_field()
        get_fun = lambda dp, up, x: f_u(dp,up[0],x)**f_v(dp,up[1],x)
        return _join_parameter_fun(get_fun, u, v)
    
    @staticmethod
    def power_scalar( u, number):
        f_u = u.discretization.get_field()
        get_fun = lambda dp, up, x: f_u(dp,up,x)**number
        return _constant_param_fun(get_fun, u)


    @staticmethod
    def apply_on_grid(fun):
        '''Returns a function applied on a grid'''
        def _f_on_grid(discr_params, field_params):
            grid = discr_params["coordinate_grid"]
            return fun(field_params, grid)
        return _f_on_grid

    def vmap_over_grid(self, fun, domain, vaxes):
        '''V-maps a function to work on a grid of values'''
        ndims = len(self.domain.N)
        for _ in range(ndims):
            fun = vmap(fun, in_axes=(None, None, 0))
        return fun

    def get_field(self):
        def f(discr_params, params, x):
            return self._get_fun(discr_params, params, x)
        return f

    def get_field_on_grid(self):
        fun = self.vmap_over_grid(
            self._get_fun,
            self.domain,
            (None, 0)
        )
        return self.apply_on_grid(fun)

    def random_field(self, seed):
        return self._init_params(seed, self.domain)

class Linear(Arbitrary):
    def __init__(self, domain):
        self.domain = domain
    
    def init_params(self, seed):
        return random.uniform(seed, self.domain.N)

    def add(self, u, v):
        discr = self.__class__(u.discretization.domain)
        @make_op(out_discr=discr, name='add')
        def _fun(u,v):
            return u+v
        return _fun(u,v)

    def add_scalar(self, u, number):
        discr = self.__class__(u.discretization.domain)
        @make_op(out_discr=discr, name='add_scalar')
        def _fun(u):
            return u+number
        out_field = _fun(u)
        return out_field

    def invert(self, u):
        discr = self.__class__(u.discretization.domain)
        @make_op(out_discr=discr, name='invert')
        def _fun(u):
            return -u
        return _fun(u)

    def mul_scalar(self, u, number):
        discr = self.__class__(u.discretization.domain)
        @make_op(out_discr=discr, name='mul_scalar')
        def _fun(u):
            return u*number
        out_field = _fun(u)
        return out_field

class GridBased(Linear):
    def __init__(self, domain):
        self.domain = domain

    def elementwise(self, u, function):
        discr = self.__class__(u.discretization.domain)
        name = f"elementwise_{function.__name__}"
        @make_op(out_discr=discr, name=name)
        def _fun(u):
            return function(u)
        return _fun(u)
    
    def mul(self, u, v):
        discr = self.__class__(u.discretization.domain)
        @make_op(out_discr=discr, name='mul')
        def _fun(u,v):
            return u+v
        return _fun(u,v)

    def power(self, u, v):
        discr = self.__class__(u.discretization.domain)
        @make_op(out_discr=discr, name='power')
        def _fun(u,v):
            return u**v
        return _fun(u,v)

    def power_scalar(self, u, number):
        discr = self.__class__(u.discretization.domain)
        @make_op(out_discr=discr, name='power_scalar')
        def _fun(u):
            return u**number
        out_field = _fun(u)
        return out_field

class FiniteDifferences(GridBased):
    """This discretization doesn't implement the `get_field()` method.
    That is because, while Lagrange polynomials or truncated series expansions
    can be used to describe continously a field in the neighbourood of a grid point,
    the interpolating function between gridded values is not uniquely defined.

    However, it can still be used to evaluate an operator as long as we only require
    the grid values.
    
    Due to the lack of `get_field()` method, this discretization cannot be mixed 
    with other discretizations (including arbitrary ones)."""
    def __init__(self, domain: Domain, dims=1, accuracy=2):
        r"""
        Args:
            domain (Domain): 
            dims (int, optional): Defaults to 1.
            accuracy (int, optional): Defaults to 2.
        """
        self.domain = domain
        # internal parameters
        self.accuracy = accuracy
        self.dims = dims
        self.is_field_complex = False

        # Initialize parameters
        self.params["coordinate_grid"] = self.domain.grid

    def derivative(self, u: Field, axis: int) -> Field:
        """Computes the derivative of a field.

        Args:
            u (Field): The field to be differentiated.
            axis (int): The axis to differentiate.

        Returns:
            Field: The derivative of the field.
        """
        # Initializes finite differences stencil and
        # adds it to the discretization parameters
        order = 1
        stencil_size = self.accuracy + 1
        stencil = self._get_stencil(
            self.accuracy, 
            self.domain.dx[axis],
            deriv_order=order
        )
        kernel_dims =  tuple([1, 1] + [stencil_size]*self.domain.ndim)
        kernel = jnp.zeros(kernel_dims)

        slices = [self.accuracy//2]*self.domain.ndim
        slices[axis]  =  slice(None)
        slices = [slice(None), 0] + slices
        slices = tuple(slices)
        kernel = kernel.at[slices].set(stencil)

        kernel_params_name = f"kernel_D1_axis_{axis}"
        self.params[kernel_params_name] = kernel

        # Generating and applying operator
        discr = self.__class__(u.discretization.domain)
        @make_op(out_discr=discr, name=f"derivative_axis_{axis}")
        def _fun(d_params, u):
            p = d_params[kernel_params_name]
            print(p)
            u = jnp.expand_dims(jnp.expand_dims(u, 0), 0) 
            return u

    def mul(self, u, v):
        discr = self.__class__(u.discretization.domain)
        @make_op(out_discr=discr, name='mul')
        def _fun(u,v):
            return u+v
        return _fun(u,v)

    @staticmethod
    def _get_analytic_stencil(accuracy, deriv_order = 1, staggered = [0,1]):
        # TODO: add reference
        """Returns a `order` accurate stencil for the second derivative."""
        assert accuracy % 2 == 0
        n_points = accuracy + 1
        
        assert deriv_order < n_points
        assert n_points % 2 == 1
        order = n_points - 1
        x, h = symbols('x, h')
        f = Function('f')
        
        dh = (h*staggered[0])/staggered[1]
        
        def TaylorExpansion(point=h, order=4, dh=0):
            return sum(point**i/factorial(i) * f(x).diff(x, i) for i in range(order+1))

        grid_points = np.arange(-(n_points-1)/2, (n_points-1)/2 + 1).astype(int)  +dh

        coef_matrix = ZeroMatrix(n_points, n_points).as_mutable()

        for p, h_coef in zip(range(n_points), grid_points):

            expansion = TaylorExpansion(h_coef * h, order, dh)

            for derivative in range(order + 1):
                term =  f(x).diff(x, derivative)
                coef_matrix[derivative, p] = expansion.coeff(term)
        
        derivative_vector = ZeroMatrix(order + 1, 1).as_mutable()
        derivative_vector[deriv_order, 0] = 1

        return coef_matrix.inv() @ derivative_vector, h

    @staticmethod
    def _get_stencil(accuracy, dx, deriv_order = 1, staggered = [0,1]):
        S, h = FiniteDifferences._get_analytic_stencil(accuracy, deriv_order, staggered)
        stencil = np.asarray(S.subs(h, dx)).astype(float)[:,0]
        return stencil

    def get_field_on_grid(self):
        r"""Returns a function `f(discretization_parameters, field_parameters)` 
        which samples the field at the grid points defined
        by `self.domain` 

        Returns:
            Callable: 
        """        
        r"""
        """
        def f(_, field_params):
            return field_params
        return f

class FourierSeries(GridBased):
    def __init__(self, domain, dims=1):
        self.domain = domain
        
        # internal parameters
        self.dims=dims
        self.is_field_complex = True

        # Initialize parameters
        # TODO: This grid could be constructed on the fly from the frequency axis or
        # stored as a parameter. Should make a speed test to check if initializing
        # is the best idea.
        self.params = {}
        self.params["freq_grid"] = self._freq_grid

    @property
    def _freq_grid(self):
        return jnp.stack(jnp.meshgrid(*self._freq_axis, indexing='ij'), axis=-1)

    @property
    def _freq_axis(self):
        f = lambda N, dx: jnp.fft.fftfreq(N,dx) * 2 * jnp.pi
        k_axis = [f(n, delta) for n, delta in zip(self.domain.N, self.domain.dx)]
        if not self.is_field_complex:
            k_axis[-1] = jnp.fft.rfftfreq(self.domain.N[-1], self.domain.dx[-1]) * 2 * jnp.pi
        return k_axis

    @property
    def _domain_axis(self):
        n_dims = len(self.domain.N)
        return list(range(-n_dims, 0))

    def get_field(self):
            N = jnp.array(self.domain.N)
            V = reduce(lambda x,y: x*y, N) # Normalization factor of FFT
            fftfun = jnp.fft.fftn if self.is_field_complex else jnp.fft.rfftn
            
            if self.is_field_complex:
                interp_fun = spectral.fft_interp
            else:
                first_dim_size = self.domain.N[0]
                interp_fun = lambda k, s, x: spectral.rfft_interp(k,s,x,first_dim_size)/V

            def f(discr_params, field_params, x):
                k = discr_params["freq_grid"]
                spectrum = fftfun(field_params, axes = self._domain_axis)
                return interp_fun(k, spectrum, x)

            return f

    def get_field_on_grid(self):
        def _sample_on_grid(_, field_params):
            return field_params
        return _sample_on_grid

    def random_field(self, rng):
        if self.is_field_complex:
            dtype = jnp.complex64
        else:
            dtype = jnp.float32
        if self.dims == 1:
            return random.normal(rng, self.domain.N, dtype) 
        else:
            return random.normal(rng, [self.dims] + [*self.domain.N], dtype)

class RealFourierSeries(FourierSeries):
    def __init__(self, domain, dims=1):
        self.domain = domain
        self.is_field_complex = False
        self.dims=dims
        self.params = {}
        self.params["freq_grid"] = self._freq_grid