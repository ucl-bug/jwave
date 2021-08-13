from jwave.geometry import Domain
from jwave.core import Discretization, Field
from jwave import primitives as pr
from jwave import spectral
from functools import reduce
from jax import random, vmap
from jax import numpy as jnp
from typing import Callable


class Arbitrary(Discretization):
    def __init__(self, domain: Domain, get_fun: Callable, init_params: Callable, dims=1):
        self._get_fun = get_fun
        self._init_params = init_params
        self.domain = domain
        self.dims=1

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

    def invert(self, u):
        return pr.Invert()(u)

    def mul(self, u, v):
        return pr.MultiplyFields()(u, v)

    def div(self, u, v):
        return pr.DivideFields()(u, v)

    def apply_on_grid(self, fun):
        """Returns a function applied on a grid"""

        def _f_on_grid(field_params):
            return fun(field_params, self.domain.grid)

        return _f_on_grid

    def vmap_over_grid(self, fun):
        """V-maps a function to work on a grid of values"""
        ndims = len(self.domain.N)
        for _ in range(ndims):
            fun = vmap(fun, in_axes=(None, 0))
        return fun

    def get_field(self):
        def f(params, x):
            return self._get_fun(params, x)

        return f

    def get_field_on_grid(self):
        fun = self.vmap_over_grid(self._get_fun)
        return self.apply_on_grid(fun)

    @staticmethod
    def div_scalar(u, scalar, independent_params=True):
        primitive = pr.DivideByScalar(
            independent_params=independent_params,
            scalar=scalar,
        )
        return primitive(u)

    @staticmethod
    def mul_scalar(u, scalar, independent_params=True):
        primitive = pr.MultiplyScalar(
            independent_params=independent_params,
            scalar=scalar,
        )
        return primitive(u)

    @staticmethod
    def power_scalar(u, scalar, independent_params=True):
        primitive = pr.PowerScalar(
            independent_params=independent_params,
            scalar=scalar,
        )
        return primitive(u)

    @staticmethod
    def gradient(u, independent_params=True):
        return pr.ArbitraryGradient()(u)

    @staticmethod
    def diag_jacobian(u, independent_params=True):
        return pr.ArbitraryDiagJacobian()(u)

    @staticmethod
    def sum_over_dims(u):
        return pr.SumOverDims()(u)

    def random_field(self, seed, name):
        params = self._init_params(seed, self.domain)
        field = Field(self, name, params)
        return params, field

    def reciprocal(self, u):
        return pr.Reciprocal()(u)


class UniformField(Arbitrary):
    def __init__(self, domain: Domain, dims=1):
        self.domain = domain
        self.dims = dims

    def random_field(self, seed):
        return random.uniform((self.dims,), seed)

    def empty_field(self):
        return jnp.zeros((self.dims,))

    def from_scalar(self, scalar, name):
        params = scalar
        field = Field(self, params, name)
        return params, field

    def get_field(self):
        def f(params, x):
            return params

        return f

    def get_field_on_grid(self):
        f = self.get_field()
        for _ in range(self.dims):
            f = vmap(f, in_axes=(None, 0))
        return f


class Coordinate(Arbitrary):
    def __init__(self, domain):
        self.domain = domain

    def init_params(self, seed, name):
        params = {}
        field = Field(self, name, params)
        return params, field

    def get_field(self):
        def f(params, x):
            return x

        return f

    def get_field_on_grid(self):
        def f(params):
            return self.domain.grid

        return f


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

    def div_scalar(self, u, scalar, independent_params=True):
        primitive = pr.DivideByScalarLinear(
            scalar=scalar, independent_params=independent_params
        )
        return primitive(u)

    def invert(self, u):
        return pr.InvertLinear()(u)

    def mul_scalar(self, u, scalar, independent_params=True):
        primitive = pr.MultiplyScalarLinear(
            scalar=scalar, independent_params=independent_params
        )
        return primitive(u)


class GridBased(Linear):
    def __init__(self, domain):
        self.domain = domain

    def elementwise(self, u, callable):
        primitive = pr.ElementwiseOnGrid(callable)
        return primitive(u)

    def power_scalar(self, u, scalar, independent_params=True):
        primitive = pr.PowerScalarLinear(
            scalar=scalar, independent_params=independent_params
        )
        return primitive(u)

    def mul(self, u, v, independent_params=True):
        primitive = pr.MultiplyOnGrid()
        return primitive(u, v)

    def div(self, u, v, independent_params=True):
        primitive = pr.DivideOnGrid()
        return primitive(u, v)

    def reciprocal(self, u):
        return pr.ReciprocalOnGrid()(u)

    def sum_over_dims(self, u):
        return pr.SumOverDimsOnGrid()(u)


class FiniteDifferences(GridBased):
    def __init__(self, domain):
        self.domain = domain
        self.is_field_complex = True


class RealFiniteDifferences(FiniteDifferences):
    def __init__(self, domain):
        super().__init__(domain)
        self.is_field_complex = False


class FourierSeries(GridBased):
    def __init__(self, domain, dims=1):
        self.domain = domain
        self.is_field_complex = True
        self.dims = dims

    @staticmethod
    def gradient(u):
        return pr.FFTGradient()(u)

    @staticmethod
    def nabla_dot(u):
        return pr.FFTNablaDot()(u)

    @staticmethod
    def diag_jacobian(u):
        return pr.FFTDiagJacobian()(u)

    @staticmethod
    def laplacian(u):
        return pr.FFTLaplacian()(u)

    @property
    def _freq_grid(self):
        return jnp.stack(jnp.meshgrid(*self._freq_axis, indexing="ij"), axis=-1)

    @property
    def _freq_axis(self):
        if self.is_field_complex:

            def f(N, dx):
                return jnp.fft.fftfreq(N, dx) * 2 * jnp.pi

        else:

            def f(N, dx):
                return jnp.fft.rfftfreq(N, dx) * 2 * jnp.pi

        k_axis = [f(n, delta) for n, delta in zip(self.domain.N, self.domain.dx)]

        return k_axis

    @property
    def _cut_freq_axis(self):
        def f(N, dx):
            return jnp.fft.fftfreq(N, dx) * 2 * jnp.pi

        k_axis = [f(n, delta) for n, delta in zip(self.domain.N, self.domain.dx)]
        if not self.is_field_complex:
            k_axis[-1] = (
                jnp.fft.rfftfreq(self.domain.N[-1], self.domain.dx[-1]) * 2 * jnp.pi
            )
        return k_axis

    @property
    def _cut_freq_grid(self):
        return jnp.stack(jnp.meshgrid(*self._cut_freq_axis, indexing="ij"), axis=-1)

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

            def interp_fun(k, s, x):
                return spectral.fft_interp(k, s, x).real

        def f(field_params, x):
            k = self._cut_freq_grid
            field_params = jnp.moveaxis(field_params, -1, 0)
            spectrum = fftfun(field_params, axes=self._domain_axis)
            interp_values = interp_fun(k, spectrum, x)
            return jnp.moveaxis(interp_values, 0, -1)

        return f

    def get_field_on_grid(self):
        def _sample_on_grid(field_params):
            return field_params

        return _sample_on_grid

    def random_field(self, seed, name):
        if self.is_field_complex:
            dtype = jnp.complex64
        else:
            dtype = jnp.float32
        params = random.normal(seed, [*self.domain.N] + [self.dims], dtype)
        field = Field(self, name, params)
        return params, field

    def empty_field(self, name):
        params = jnp.zeros([*self.domain.N] + [self.dims])
        if self.is_field_complex:
            params = params + 0j
        field = Field(self, name, params)
        return params, field


class RealFourierSeries(FourierSeries):
    def __init__(self, domain, dims=1):
        self.domain = domain
        self.is_field_complex = False
        self.dims = dims

    @staticmethod
    def gradient(u):
        return pr.FFTGradient(real=True)(u)

    @staticmethod
    def nabla_dot(u):
        return pr.FFTNablaDot(real=True)(u)

    @staticmethod
    def diag_jacobian(u):
        return pr.FFTDiagJacobian(real=True)(u)


class StaggeredRealFourier(RealFourierSeries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def staggered_grad(u, c_ref, dt, direction):
        return pr.FFTStaggeredGrad(c_ref=c_ref, dt=dt, direction=direction)(u)

    @staticmethod
    def staggered_diag_jacobian(u, c_ref, dt, direction):
        return pr.FFTStaggeredDiagJacobian(c_ref=c_ref, dt=dt, direction=direction)(u)
