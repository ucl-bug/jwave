from jwave.core import TracedField, Discretization
from jwave import discretization
from typing import Callable, NamedTuple, List, Any
from jax import random

def no_init(*args, **kwargs):
    raise RuntimeError(
        "Can't initialize a field resulting from applying an operator"
    )

class Operator(NamedTuple):
    '''A concrete operation of the computational graph'''
    fun: Callable
    inputs: List[str]
    params: Any
    param_kind: str
    yelds: TracedField

    def __repr__(self):
        keys = tuple(self.inputs)
        return f"{self.yelds.name}: {self.yelds.discretization} <-- {self.fun.__name__} {keys} | ({self.param_kind}) {self.params}"

class Primitive(object):
    def __init__(self, name=None, independent_params=True):
        self.name = name
        self.independent_params = independent_params

    def discrete_transform(self):
        '''To be implemented in childs'''
        return Callable
    
    def setup(self, field):
        '''To be implemented in childs'''
        return dict, Discretization

    def __call__(self, field):
        '''Returns an operation stored in the computational graph'''
        tracer = field.tracer
        counter = tracer.counter()

        # Initialize transformation parameter and out discretization
        primitive_parameters, output_discretization = self.setup(field)

        # Adds the parameters to the globals
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
        '''New arbitrary discretization'''
        parameters = {"scalar": self.scalar}

        def get_field(p_joined, x):
            p, scalar = p_joined
            return field.discretization.get_field()(p,x) + scalar

        new_discretization = discretization.Arbitrary(
            field.discretization.domain,
            get_field,
            no_init
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
        '''Same discretization family as the input'''
        new_discretization = field.discretization
        parameters = {"scalar": self.scalar}
        return parameters, new_discretization