from jwave.geometry import Domain
from jwave.core import Discretization, Field
from jwave import primitives as pr
from jwave import spectral
from functools import reduce
from jax import random, vmap
from jax import numpy as jnp
from typing import Callable
from sympy import symbols, Function, factorial, ZeroMatrix
import numpy as np


class Arbitrary(Discretization):
    def __init__(self, domain: Domain, get_fun: Callable, init_params: Callable):
        self._get_fun = get_fun
        self._init_params = init_params
        self.domain = domain

    @staticmethod
    def add_scalar(u, scalar, independent_params=True):
        primitive = pr.AddScalar(
            independent_params=independent_params,
            scalar=scalar,
        )
        return primitive(u)

    def add(self, u, v, independent_params=True):
        primitive = pr.AddField()
        return primitive(u, v)

    def elementwise(self, u, callable):
        primitive = pr.Elementwise(callable)
        return primitive(u)

    def apply_on_grid(self, fun):
        """Returns a function applied on a grid"""

        def _f_on_grid(field_params):
            return fun(field_params, self.domain.grid)

        return _f_on_grid

    def vmap_over_grid(self, fun):
        """V-maps a function to work on a grid of values"""
        ndims = len(self.domain.N)
        for _ in range(ndims):
            fun = vmap(fun, in_axes=(None, None, 0))
        return fun

    def get_field(self):
        def f(params, x):
            return self._get_fun(params, x)

        return f

    def get_field_on_grid(self):
        fun = self.vmap_over_grid(self._get_fun, self.domain, (None, 0))
        return self.apply_on_grid(fun)

    def random_field(self, seed):
        return self._init_params(seed, self.domain)


class Linear(Arbitrary):
    def __init__(self, domain):
        self.domain = domain

    def init_params(self, seed):
        return random.uniform(seed, self.domain.N)

    def add_scalar(self, u, scalar, independent_params=True):
        primitive = pr.AddScalarLinear(
            scalar=scalar, independent_params=independent_params
        )
        return primitive(u)

    def add(self, u, v, independent_params=True):
        primitive = pr.AddFieldLinearSame()
        return primitive(u, v)


class GridBased(Linear):
    def __init__(self, domain):
        self.domain = domain

    def elementwise(self, u, callable):
        primitive = pr.ElementwiseOnGrid(callable)
        return primitive(u)


class FourierSeries(GridBased):
    def __init__(self, domain, dims=1):
        self.domain = domain

        # internal parameters
        self.dims = dims
        self.is_field_complex = True

        # Initialize parameters
        # TODO: This grid could be constructed on the fly from the frequency axis or
        # stored as a parameter. Should make a speed test to check if initializing
        # is the best idea.
        self.params = {}
        self.params["freq_grid"] = self._freq_grid

    @property
    def _freq_grid(self):
        return jnp.stack(jnp.meshgrid(*self._freq_axis, indexing="ij"), axis=-1)

    @property
    def _freq_axis(self):
        f = lambda N, dx: jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
        k_axis = [f(n, delta) for n, delta in zip(self.domain.N, self.domain.dx)]
        if not self.is_field_complex:
            k_axis[-1] = (
                jnp.fft.rfftfreq(self.domain.N[-1], self.domain.dx[-1]) * 2 * jnp.pi
            )
        return k_axis

    @property
    def _domain_axis(self):
        n_dims = len(self.domain.N)
        return list(range(-n_dims, 0))

    def get_field(self):
        N = jnp.array(self.domain.N)
        V = reduce(lambda x, y: x * y, N)  # Normalization factor of FFT
        fftfun = jnp.fft.fftn if self.is_field_complex else jnp.fft.rfftn

        if self.is_field_complex:
            interp_fun = spectral.fft_interp
        else:
            first_dim_size = self.domain.N[0]
            interp_fun = (
                lambda k, s, x: spectral.rfft_interp(k, s, x, first_dim_size) / V
            )

        def f(field_params, x):
            k = self._freq_grid
            spectrum = fftfun(field_params, axes=self._domain_axis)
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
        self.dims = dims
        self.params = {}
        self.params["freq_grid"] = self._freq_grid
