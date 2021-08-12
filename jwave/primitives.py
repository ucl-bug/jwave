from jwave.core import TracedField, Discretization
from jwave import discretization, geometry
from typing import Callable, NamedTuple, List, Any
from jax import numpy as jnp


def no_init(*args, **kwargs):
    raise RuntimeError("Can't initialize a field resulting from applying an operator")


class Operator(NamedTuple):
    """A concrete operation of the computational graph"""

    fun: Callable
    inputs: List[str]
    params: Any
    param_kind: str
    yelds: TracedField

    def __repr__(self):
        keys = tuple(self.inputs)
        return (
            f"{self.yelds.name}: {self.yelds.discretization} "
            + f"<-- {self.fun.__name__} {keys} | ({self.param_kind}) {self.params}"
        )


class Primitive(object):
    def __init__(self, name=None, independent_params=True):
        self.name = name
        self.independent_params = independent_params

    def discrete_transform(self):
        """To be implemented in childs"""
        return Callable

    def setup(self, field):
        """To be implemented in childs"""
        return dict, Discretization

    def __call__(self, field):
        """Returns an operation stored in the computational graph"""
        tracer = field.tracer
        counter = tracer.counter()

        # Initialize transformation parameter and out discretization
        primitive_parameters, output_discretization = self.setup(field)

        # Adds the parameters to the globals
        if primitive_parameters is None:
            params = {}
            name = self.name
            param_kind = "none"
        else:
            if self.independent_params:
                name = f"{self.name}_{counter}"
                tracer.globals.set(name, primitive_parameters, "independent")
                param_kind = "independent"
                params = primitive_parameters
            else:
                name = self.name
                tracer.globals.set(name, primitive_parameters, "shared")
                param_kind = "shared"
                params = primitive_parameters

        # Extract discrete transform parameters
        fun = self.discrete_transform()

        # Construct operator compatible with computational graph
        args = [field.name]

        # Adds primitive to the tracer and returns output field
        u_name = f"_{tracer.counter()}"
        outfield = TracedField(output_discretization, params, tracer, u_name)
        op = Operator(fun, args, name, param_kind, outfield)
        tracer.add_operation(op)

        return outfield


class BinaryPrimitive(Primitive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, field_1, field_2):
        tracer = field_1.tracer
        counter = tracer.counter()

        # Initialize transformation parameter and out discretization
        primitive_parameters, output_discretization = self.setup(field_1, field_2)

        # Adds the parameters to the globals
        if primitive_parameters is None:
            params = {}
            name = self.name
            param_kind = "none"
        else:
            if self.independent_params:
                name = f"{self.name}_{counter}"
                tracer.globals.set(name, primitive_parameters, "independent")
                param_kind = "independent"
                params = primitive_parameters
            else:
                name = self.name
                tracer.globals.set(name, primitive_parameters, "shared")
                param_kind = "shared"
                params = primitive_parameters

        # Extract discrete transform parameters
        fun = self.discrete_transform()

        # Construct operator compatible with computational graph
        args = [field_1.name, field_2.name]

        # Adds primitive to the tracer and returns output field
        u_name = f"_{tracer.counter()}"
        outfield = TracedField(output_discretization, params, tracer, u_name)
        op = Operator(fun, args, name, param_kind, outfield)
        tracer.add_operation(op)

        return outfield


class AddScalar(Primitive):
    def __init__(self, scalar, name="AddScalar", independent_params=True):
        super().__init__(name, independent_params)
        self.scalar = scalar

    def discrete_transform(self):
        def f(op_params, field_params):
            return [field_params, op_params["scalar"]]

        f.__name__ = self.name
        return f

    def setup(self, field):
        """New arbitrary discretization"""
        parameters = {"scalar": self.scalar}

        def get_field(p_joined, x):
            p, scalar = p_joined
            return field.discretization.get_field()(p, x) + scalar

        new_discretization = discretization.Arbitrary(
            field.discretization.domain, get_field, no_init
        )

        return parameters, new_discretization


class AddScalarLinear(Primitive):
    def __init__(self, scalar, name="AddScalarLinear", independent_params=True):
        super().__init__(name, independent_params)
        self.scalar = scalar

    def discrete_transform(self):
        def f(op_params, field_params):
            return field_params + op_params["scalar"]

        f.__name__ = self.name
        return f

    def setup(self, field):
        """Same discretization family as the input"""
        new_discretization = field.discretization
        parameters = {"scalar": self.scalar}
        return parameters, new_discretization


class AddField(BinaryPrimitive):
    def __init__(self, name="AddField", independent_params=True):
        super().__init__(name, independent_params)

    def discrete_transform(self):
        def f(op_params, field_1_params, field_2_params):
            return [field_1_params, field_2_params]

        f.__name__ = self.name
        return f

    def setup(self, field_1, field_2):
        # Must have the same domain
        assert field_1.discretization.domain == field_2.discretization.domain

        def get_field(p_joined, x):
            p1, p2 = p_joined
            return field_1.discretization.get_field()(
                p1, x
            ) + field_2.discretization.get_field()(p2, x)

        new_discretization = discretization.Arbitrary(
            field_1.discretization.domain, get_field, no_init
        )

        return None, new_discretization


class AddFieldLinearSame(BinaryPrimitive):
    def __init__(self, name="AddFieldLinearSame", independent_params=True):
        super().__init__(name, independent_params)

    def discrete_transform(self):
        def f(op_params, field_1_params, field_2_params):
            return field_1_params + field_2_params

        f.__name__ = self.name
        return f

    def setup(self, field_1, field_2):
        assert field_1.discretization.domain == field_2.discretization.domain
        assert isinstance(field_1.discretization, type(field_2.discretization))

        new_discretization = field_1.discretization
        return None, new_discretization


class MultiplyFields(BinaryPrimitive):
    def __init__(self, name="MultiplyFields", independent_params=True):
        super().__init__(name, independent_params)

    def discrete_transform(self):
        def f(op_params, field_1_params, field_2_params):
            return [field_1_params, field_2_params]

        f.__name__ = self.name
        return f

    def setup(self, field_1, field_2):
        def get_field(p_joined, x):
            p1, p2 = p_joined
            return field_1.discretization.get_field()(
                p1, x
            ) * field_2.discretization.get_field()(p2, x)

        new_discretization = discretization.Arbitrary(
            field_1.discretization.domain, get_field, no_init
        )
        return None, new_discretization


class DivideFields(BinaryPrimitive):
    def __init__(self, name="DivideFields", independent_params=True):
        super().__init__(name, independent_params)

    def discrete_transform(self):
        def f(op_params, field_1_params, field_2_params):
            return [field_1_params, field_2_params]

        f.__name__ = self.name
        return f

    def setup(self, field_1, field_2):
        def get_field(p_joined, x):
            p1, p2 = p_joined
            return field_1.discretization.get_field()(
                p1, x
            ) / field_2.discretization.get_field()(p2, x)

        new_discretization = discretization.Arbitrary(
            field_1.discretization.domain, get_field, no_init
        )
        return None, new_discretization


class MultiplyOnGrid(BinaryPrimitive):
    def __init__(self, name="MultiplyOnGrid", independent_params=True):
        super().__init__(name, independent_params)

    def discrete_transform(self):
        def f(op_params, field_1_params, field_2_params):
            return field_1_params * field_2_params

        f.__name__ = self.name
        return f

    def setup(self, field_1, field_2):
        assert field_1.discretization.domain == field_2.discretization.domain
        assert isinstance(field_1.discretization, type(field_2.discretization))

        new_discretization = field_1.discretization
        return None, new_discretization


class DivideOnGrid(BinaryPrimitive):
    def __init__(self, name="DivideOnGrid", independent_params=True):
        super().__init__(name, independent_params)

    def discrete_transform(self):
        def f(op_params, field_1_params, field_2_params):
            return field_1_params / field_2_params

        f.__name__ = self.name
        return f

    def setup(self, field_1, field_2):
        assert field_1.discretization.domain == field_2.discretization.domain
        assert isinstance(field_1.discretization, type(field_2.discretization))

        new_discretization = field_1.discretization
        return None, new_discretization


class SumOverDimsOnGrid(Primitive):
    def __init__(self, name="SumOverDimsOnGrid", independent_params=True):
        super().__init__(name, independent_params)

    def discrete_transform(self):
        def f(op_params, field_params):
            return jnp.sum(field_params, axis=-1, keepdims=True)

        f.__name__ = self.name
        return f

    def setup(self, field):
        new_discretization = field.discretization
        return None, new_discretization


class Elementwise(Primitive):
    def __init__(self, callable, name="Elementwise", independent_params=True):
        super().__init__(name, independent_params)
        self.callable = callable

    def discrete_transform(self):
        def f(op_params, field_params):
            return field_params

        f.__name__ = self.name
        return f

    def setup(self, field):
        """New arbitrary discretization"""

        def get_field(p, x):
            return self.callable(field.discretization.get_field()(p, x))

        new_discretization = discretization.Arbitrary(
            field.discretization.domain, get_field, no_init
        )

        return None, new_discretization


class ElementwiseOnGrid(Primitive):
    def __init__(self, callable, name="ElementwiseOnGrid", independent_params=True):
        super().__init__(name, independent_params)
        self.callable = callable

    def discrete_transform(self):
        def f(op_params, field_params):
            return self.callable(field_params)

        f.__name__ = self.name
        return f

    def setup(self, field_1):
        new_discretization = field_1.discretization
        return None, new_discretization


class DivideByScalar(Primitive):
    def __init__(self, scalar, name="DivideByScalar", independent_params=True):
        super().__init__(name, independent_params)
        self.scalar = scalar

    def discrete_transform(self):
        def f(op_params, field_params):
            return [field_params, op_params["scalar"]]

        f.__name__ = self.name
        return f

    def setup(self, field):
        """New arbitrary discretization"""
        parameters = {"scalar": self.scalar}

        def get_field(p_joined, x):
            p, scalar = p_joined
            return field.discretization.get_field()(p, x) / scalar

        new_discretization = discretization.Arbitrary(
            field.discretization.domain, get_field, no_init
        )

        return parameters, new_discretization


class DivideByScalarLinear(Primitive):
    def __init__(self, scalar, name="DivideByScalarLinear", independent_params=True):
        super().__init__(name, independent_params)
        self.scalar = scalar

    def discrete_transform(self):
        def f(op_params, field_params):
            return field_params / op_params["scalar"]

        f.__name__ = self.name
        return f

    def setup(self, field):
        """Same discretization family as the input"""
        new_discretization = field.discretization
        parameters = {"scalar": self.scalar}
        return parameters, new_discretization


class FFTGradient(Primitive):
    def __init__(self, real=False, name="FFTGradient", independent_params=False):
        super().__init__(name, independent_params)
        self.real = real

    def setup(self, field):
        new_discretization = field.discretization
        k_vec = field.discretization._freq_axis
        parameters = {"k_vec": k_vec}
        return parameters, new_discretization

    def discrete_transform(self):
        if self.real:
            ffts = [jnp.fft.rfft, jnp.fft.irfft]
        else:
            ffts = [jnp.fft.fft, jnp.fft.ifft]

        def f(op_params, field_params):
            k_vec = op_params["k_vec"]
            u = field_params[..., 0]
            ndim = len(field_params.shape) - 1

            def single_grad(axis, u):
                u = jnp.moveaxis(u, axis, -1)
                Fx = ffts[0](u, axis=-1)
                iku = 1j * Fx * k_vec[axis]
                du = ffts[1](iku, axis=-1, n=u.shape[-1])
                return jnp.moveaxis(du, -1, axis)

            return jnp.stack([single_grad(i, u) for i in range(ndim)], axis=-1)

        f.__name__ = self.name

        return f


class FFTLaplacian(Primitive):
    def __init__(self, real=False, name="FFTLaplacian", independent_params=False):
        super().__init__(name, independent_params)
        self.real = real

    def setup(self, field):
        new_discretization = field.discretization
        k_vec = field.discretization._freq_axis
        parameters = {"k_vec": k_vec}
        return parameters, new_discretization

    def discrete_transform(self):
        if self.real:
            ffts = [jnp.fft.rfft, jnp.fft.irfft]
        else:
            ffts = [jnp.fft.fft, jnp.fft.ifft]

        def f(op_params, field_params):
            k_vec = op_params["k_vec"]
            u = field_params[..., 0]
            ndim = len(field_params.shape) - 1

            def single_grad(axis, u):
                u = jnp.moveaxis(u, axis, -1)
                Fx = ffts[0](u, axis=-1)
                iku = -Fx * k_vec[axis] ** 2
                du = ffts[1](iku, axis=-1, n=u.shape[-1])
                return jnp.moveaxis(du, -1, axis)

            return jnp.sum(
                jnp.stack([single_grad(i, u) for i in range(ndim)], axis=-1),
                axis=-1,
                keepdims=True,
            )

        f.__name__ = self.name

        return f


class FFTNablaDot(Primitive):
    def __init__(self, name="FFTNablaDot", independent_params=False):
        super().__init__(name, independent_params)

    def setup(self, field):
        new_discretization = field.discretization
        k_vec = field.discretization._freq_axis
        parameters = {"k_vec": k_vec}
        return parameters, new_discretization

    def discrete_transform(self):
        def f(op_params, field_params):
            k_vec = op_params["k_vec"]
            ndim = len(field_params.shape) - 1

            res = jnp.zeros_like(field_params[..., 0])

            def single_grad(axis, u):
                u = jnp.moveaxis(u, axis, -1)
                Fx = jnp.fft.fft(u, axis=-1)
                iku = 1j * Fx * k_vec[axis]
                du = jnp.fft.ifft(iku, axis=-1)
                return jnp.moveaxis(du, -1, axis)

            for ax in range(ndim):
                res = res + single_grad(ax, field_params[..., ax])
            return res

        f.__name__ = self.name
        return f


class FFTDiagJacobian(Primitive):
    def __init__(self, real=False, name="FFTDiagJacobian", independent_params=False):
        super().__init__(name, independent_params)
        self.real = real

    def setup(self, field):
        new_discretization = field.discretization
        k_vec = field.discretization._freq_axis
        parameters = {"k_vec": k_vec}
        return parameters, new_discretization

    def discrete_transform(self):
        if self.real:
            ffts = [jnp.fft.rfft, jnp.fft.irfft]
        else:
            ffts = [jnp.fft.fft, jnp.fft.ifft]

        def f(op_params, field_params):
            k_vec = op_params["k_vec"]
            ndim = len(field_params.shape) - 1

            res = jnp.zeros_like(field_params)

            def single_grad(axis, u):
                u = jnp.moveaxis(u, axis, -1)
                Fx = ffts[0](u, axis=-1)
                iku = 1j * Fx * k_vec[axis]
                du = ffts[1](iku, axis=-1, n=u.shape[-1])
                return jnp.moveaxis(du, -1, axis)

            for ax in range(ndim):
                res = res.at[..., ax].set(single_grad(ax, field_params[..., ax]))
            return res

        f.__name__ = self.name
        return f


class FFTStaggeredGrad(Primitive):
    def __init__(
        self,
        name="FFTStaggeredGrad",
        c_ref=1.0,
        dt=1.0,
        direction=geometry.Staggered.NONE,
        independent_params=False,
    ):
        super().__init__(name, independent_params)
        self.direction = direction
        self.c_ref = c_ref
        self.dt = dt

    def setup(self, field):
        def f(N, dx):
            return jnp.fft.fftfreq(N, dx) * 2 * jnp.pi

        domain = field.discretization.domain
        k_vec = [f(n, delta) for n, delta in zip(domain.N, domain.dx)]

        new_discretization = field.discretization
        dx = field.discretization.domain.dx

        K = jnp.stack(jnp.meshgrid(*k_vec, indexing="ij"))
        k_magnitude = jnp.sqrt(jnp.sum(K ** 2, 0))
        k_space_op = jnp.expand_dims(
            jnp.sinc(self.c_ref * k_magnitude * self.dt / (2 * jnp.pi)), -1
        )

        parameters = {"k_vec": k_vec, "dx": dx, "k_space_op": k_space_op}
        return parameters, new_discretization

    def discrete_transform(self):
        def f(op_params, field_params):
            # Extract parameters for convenience
            k_vec = op_params["k_vec"]
            dx = op_params["dx"]
            kspaceop = op_params["k_space_op"]

            # Make the modified k-vectors
            k_vec = [
                1j * k * jnp.exp(1j * k * self.direction * delta / 2)
                for k, delta in zip(k_vec, dx)
            ]

            # Perform directional gradients with a single Forward FFT
            # NOTE: This could have done better with are real FFT
            P = jnp.fft.fftn(field_params)

            def make_dx(FP, axis):
                FdP = jnp.moveaxis(jnp.moveaxis(FP, axis, -1) * k_vec[axis], -1, axis)
                return jnp.fft.ifftn(FdP * kspaceop).real

            ndim = len(field_params.shape) - 1
            dp = jnp.concatenate([make_dx(P, ax) for ax in range(ndim)], axis=-1)

            return dp

        f.__name__ = self.name
        return f


class FFTStaggeredDiagJacobian(FFTStaggeredGrad):
    def __init__(
        self,
        name="FFTStaggeredDiagJacobian",
        c_ref=1.0,
        dt=1.0,
        direction=geometry.Staggered.NONE,
        independent_params=False,
    ):
        super().__init__(name, independent_params)
        self.direction = direction
        self.c_ref = c_ref
        self.dt = dt

    def discrete_transform(self):
        def f(op_params, field_params):
            # Extract parameters for convenience
            k_vec = op_params["k_vec"]
            dx = op_params["dx"]
            kspaceop = op_params["k_space_op"]

            # Make the modified k-vectors
            k_vec = [
                1j * k * jnp.exp(1j * k * self.direction * delta / 2)
                for k, delta in zip(k_vec, dx)
            ]

            # Perform directional gradients with a single Forward FFT
            # NOTE: This could have done better with are real FFT
            def make_dx(P, axis):
                FP = jnp.fft.fftn(P)
                FdP = jnp.moveaxis(jnp.moveaxis(FP, axis, -1) * k_vec[axis], -1, axis)
                output = jnp.fft.ifftn(FdP * kspaceop).real
                return output

            ndim = len(field_params.shape) - 1
            dp = jnp.concatenate(
                [make_dx(field_params[..., ax], ax) for ax in range(ndim)], axis=-1
            )
            return dp

        f.__name__ = self.name
        return f


class Reciprocal(Primitive):
    def __init__(self, name="Reciprocal", independent_params=True):
        super().__init__(name, independent_params)

    def discrete_transform(self):
        def f(op_params, field_params):
            return field_params

        f.__name__ = self.name
        return f

    def setup(self, field):
        """New arbitrary discretization"""

        def get_field(p, x):
            return 1.0 / field.discretization.get_field()(p, x)

        new_discretization = discretization.Arbitrary(
            field.discretization.domain, get_field, no_init
        )

        return None, new_discretization


class Invert(Primitive):
    def __init__(self, name="Invert", independent_params=True):
        super().__init__(name, independent_params)

    def discrete_transform(self):
        def f(op_params, field_params):
            return field_params

        f.__name__ = self.name
        return f

    def setup(self, field):
        """New arbitrary discretization"""

        def get_field(p, x):
            return 1.0 / field.discretization.get_field()(p, x)

        new_discretization = discretization.Arbitrary(
            field.discretization.domain, get_field, no_init
        )
        return None, new_discretization


class InvertLinear(Primitive):
    def __init__(self, name="InvertLinear", independent_params=True):
        super().__init__(name, independent_params)

    def discrete_transform(self):
        def f(op_params, field_params):
            return -field_params

        f.__name__ = self.name
        return f

    def setup(self, field):
        """Same discretization family as the input"""
        new_discretization = field.discretization
        return None, new_discretization


class ReciprocalOnGrid(Primitive):
    def __init__(self, name="ReciprocalOnGrid", independent_params=True):
        super().__init__(name, independent_params)

    def discrete_transform(self):
        def f(op_params, field_params):
            return 1.0 / field_params

        f.__name__ = self.name
        return f

    def setup(self, field):
        """Same discretization family as the input"""
        new_discretization = field.discretization
        return None, new_discretization


class MultiplyScalar(Primitive):
    def __init__(self, scalar, name="MultiplyScalar", independent_params=True):
        super().__init__(name, independent_params)
        self.scalar = scalar

    def discrete_transform(self):
        def f(op_params, field_params):
            return [field_params, op_params["scalar"]]

        f.__name__ = self.name
        return f

    def setup(self, field):
        """New arbitrary discretization"""
        parameters = {"scalar": self.scalar}

        def get_field(p_joined, x):
            p, scalar = p_joined
            return field.discretization.get_field()(p, x) * scalar

        new_discretization = discretization.Arbitrary(
            field.discretization.domain, get_field, no_init
        )

        return parameters, new_discretization


class MultiplyScalarLinear(Primitive):
    def __init__(self, scalar, name="MultiplyScalarLinear", independent_params=True):
        super().__init__(name, independent_params)
        self.scalar = scalar

    def discrete_transform(self):
        def f(op_params, field_params):
            return field_params * op_params["scalar"]

        f.__name__ = self.name
        return f

    def setup(self, field):
        """Same discretization family as the input"""
        new_discretization = field.discretization
        parameters = {"scalar": self.scalar}
        return parameters, new_discretization


class PowerScalar(Primitive):
    def __init__(self, scalar, name="PowerScalar", independent_params=True):
        super().__init__(name, independent_params)
        self.scalar = scalar

    def discrete_transform(self):
        def f(op_params, field_params):
            return [field_params, op_params["scalar"]]

        f.__name__ = self.name
        return f

    def setup(self, field):
        """New arbitrary discretization"""
        parameters = {"scalar": self.scalar}

        def get_field(p_joined, x):
            p, scalar = p_joined
            return field.discretization.get_field()(p, x) ** scalar

        new_discretization = discretization.Arbitrary(
            field.discretization.domain, get_field, no_init
        )

        return parameters, new_discretization


class PowerScalarLinear(Primitive):
    def __init__(self, scalar, name="PowerScalarLinear", independent_params=True):
        super().__init__(name, independent_params)
        self.scalar = scalar

    def discrete_transform(self):
        def f(op_params, field_params):
            return field_params ** op_params["scalar"]

        f.__name__ = self.name
        return f

    def setup(self, field):
        """Same discretization family as the input"""
        new_discretization = field.discretization
        parameters = {"scalar": self.scalar}
        return parameters, new_discretization
