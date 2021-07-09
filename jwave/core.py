from jwave.geometry import Domain
from jwave import operators as jops
from typing import Callable, NamedTuple, List
from collections.abc import Iterable
from functools import wraps
from hashids import Hashids

class Discretization(object):
    domain: Domain
    params: dict = {}

    def init_params(self, seed):
        return 0

    def __repr__(self):
        return self.__class__.__name__

class Tracer(object):
    def __init__(self):
        self.input_fields= {}
        self.discretization_params= {}
        self.operations= {}
        self._counter = 0
        self._hash = Hashids()

    def counter(self):
        self._counter += 1
        return self._hash.encode(self._counter)

    def add_input_field(self, name, field):
        if name not in self.input_fields.keys():
            self.input_fields[name] = field

    def add_operation(self, op):
        self.operations[op.yelds.name] = op

    def add_discretization(self, discr):
        for param in discr.params.keys():
            if param not in self.discretization_params.keys():
                self.discretization_params[param] = discr.params[param]

    def construct_function(self, sorted_graph: dict, names: List[str]):
        concrete_ops = {}

        # Materialize constants
        for op in self.operations.keys():
            const = [k for k in self.operations[op].const.keys()]
            for c in const:
                if c in sorted_graph:
                    concrete_ops[c] = self.operations[op].const[c]

        def f(discrete_params, field_params):
            # Add used inputs
            for field_name in field_params.keys():
                if field_name in sorted_graph:
                    concrete_ops[field_name] = field_params[field_name]

            # Add used discretization parameters
            for d_param in self.discretization_params:
                if d_param in sorted_graph:
                    concrete_ops[d_param] = discrete_params[d_param]

            # Compose operations present in the sorted grap
            for op_name in self.operations.keys():
                if op_name in sorted_graph:
                    op = self.operations[op_name]
                    args = [concrete_ops[n] for n in op.args]
                    concrete_ops[op_name] = op.fun(*args)

            # Output requested parameters
            outputs = []
            for n in names:
                outputs.append(concrete_ops[n])
            return outputs
        return f

    @staticmethod
    def _filter_unneded_ops(operations: dict, out_fields: List[str]):
        op_keys = [k for k in operations.keys()]
        op_args = [k.args for k in operations.values()]
        stack = out_fields
        out_set = []
        while stack:    
            x = stack.pop()
            out_set.append(x)
            # if is not an input or constant
            if x in op_keys:
                for x in op_args[op_keys.index(x)]:
                    if x not in out_set:
                        stack.append(x)
        return set(out_set)

    def __repr__(self):
        ops = ["- " + str(x)+"\n" for x in self.operations.values()]
        ops = ''.join(ops)
        input_keys = tuple([*self.input_fields.keys()])
        discr_keys = tuple([*self.discretization_params.keys()])
        string = f"Input fields: {input_keys}\n\n"
        string += f"Discretization parameters: {discr_keys}\n\n"
        string += f"Operations:\n{ops}"
        return string

class Field(object):
    def __init__(self, discretization, name, params):
        self.params=params
        self.discretization=discretization
        self.name=name

    def init_params(self, seed):
        return self.discretization.init_params(seed)

    def __add__(self, other):
        if type(other) == int or type(other) == float:
            return jops.add_scalar(self, other)
        else:
            return jops.add(self, other)

    def __mul__(self, other):
        if type(other) == int or type(other) == float:
            return jops.mul_scalar(self, other)
        else:
            return jops.mul(self, other)

    def __pow__(self, other):
        if type(other) == int or type(other) == float:
            return jops.power_scalar(self, other)
        else:
            return jops.power(self, other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        self = jops.invert(self)
        return self + other

    def __sub__(self, other):
        other = jops.invert(other)
        return self + other

    def __repr__(self):
        return f"Field :: {self.discretization}, {self.name}"

class TracedField(Field):
    def __init__(
        self, 
        discretization: Discretization, 
        params: dict,
        tracer: Tracer,
        name: str
    ):
        self.params = params
        self.discretization = discretization
        self.tracer = tracer
        self.name=name
        self.preprocess_params = lambda x,y: y # Identity function for parameters

    @staticmethod
    def from_field(field, tracer, name):
        return TracedField(field.discretization, field.params, tracer, name)

    def __repr__(self):
        return f"-- TracedField :: {self.discretization}, {self.name} \n\n{self.tracer}"

    def get_field(self):
        f = self.discretization.get_field()
        def wrapped_f(discr_params, input_params, x):
            new_params = self.preprocess_params(discr_params, input_params)
            return f(discr_params, new_params, x)
        return wrapped_f

    def get_field_on_grid(self):
        f = self.discretization.get_field_on_grid()
        def wrapped_f(discr_params, input_params):
            new_params = self.preprocess_params(discr_params, input_params)
            return f(discr_params, new_params)
        return wrapped_f

class Operation(NamedTuple):
    '''A concrete operation of the computational graph'''
    fun: Callable
    args: dict
    const: dict
    yelds: TracedField

    def __repr__(self):
        keys = tuple(self.args)
        constants = ["{}: {}".format(a,b) for a,b in zip(self.const.keys(), self.const.values())]
        if len(constants) > 0:
            return f"{self.yelds.name}: {self.yelds.discretization} <-- {self.fun.__name__} {keys} | {constants}"
        else:
            return f"{self.yelds.name}: {self.yelds.discretization} <-- {self.fun.__name__} {keys}"

class DiscretizedOperator(object):
    def __init__(self, discr_params, preprocess_func, fields, global_preprocess, tracer):
        self.discr_params = discr_params
        self.preprocess_func = preprocess_func
        self.fields = fields
        self.global_preprocess = global_preprocess
        self.tracer = tracer

    def __repr__(self):
        discr = [x.discretization for x in self.fields]
        names = [x.name for x in self.fields]
        return f"DiscretizedOperator :: {discr}, {names} \n\n {self.tracer}"

    def get_field(self, idx=-1):
        '''idx=-1 means all fields'''
        if idx != -1:
            f = self.fields[idx].discretization.get_field()
            preprocess = self.preprocess_func[idx]
            def wrapped_f(discr_params, input_params, x):
                new_params = preprocess(discr_params, input_params)
                return f(discr_params, new_params, x)
            return wrapped_f
        else:
            preprocess = self.global_preprocess
            f_all = [f.discretization.get_field() for f in self.fields]
            def wrapped_f(discr_params, input_params, x):
                all_new_params = preprocess(discr_params, input_params)
                return [f(discr_params, all_new_params[i], x) for i,f in enumerate(f_all)]
            return wrapped_f

    def get_field_on_grid(self, idx=-1):
        '''idx=-1 means all fields'''
        if idx != -1:
            f = self.fields[idx].discretization.get_field_on_grid()
            preprocess = self.preprocess_func[idx]
            def wrapped_f(discr_params, input_params):
                new_params = preprocess(discr_params, input_params)
                return f(discr_params, new_params)
            return wrapped_f
        else:
            preprocess = self.global_preprocess
            f_all = [f.discretization.get_field_on_grid() for f in self.fields]
            def wrapped_f(discr_params, input_params):
                all_new_params = preprocess(discr_params, input_params)
                return [f(discr_params, all_new_params[i]) for i,f in enumerate(f_all)]
            return wrapped_f

def operator(has_aux=False, debug=False):
    '''Returns a decorator that builds the computational graph of an operator'''
    if has_aux == True:
        raise NotImplementedError("Operator with auxiliary arguments are not currently supported.")
    
    def decorator(fun):
        def wrapper(*args, **kwargs):

            tracer = Tracer()

            # Use only named variables
            if len(args) > 0:
                raise ValueError("Only keyword arguments can be used in a traced function")
            
            # Register input fields
            for input_name in kwargs.keys():
                field = TracedField.from_field(kwargs[input_name], tracer, name=input_name)
                tracer.add_input_field(name=input_name, field=field)

            # Register operations
            out_fields = fun(*[x for x in tracer.input_fields.values()])
            if not isinstance(out_fields, Iterable):
                out_fields = (out_fields,)
            
            # Register discretization parameters
            for f in out_fields:
                tracer.add_discretization(f.discretization)
            
            # Generates jax function for each output variable
            out_names = [x.name for x in out_fields]
            p_func = []
            fields = []
            for out in out_names:
                sorted_graph = tracer._filter_unneded_ops(tracer.operations, [out])
                if debug:
                    print(f'Sorted graph for {out}:\n{sorted_graph}\n')
                pf = tracer.construct_function(sorted_graph, [out])
                preprocess_func = lambda x,y: pf(x,y)[0] # Only need one output
                field = tracer.operations[out].yelds

                p_func.append(preprocess_func)
                fields.append(field)

            # Making function preprocessing parameters for all fields
            sorted_graph = tracer._filter_unneded_ops(tracer.operations, out_names)
            global_preprocess = tracer.construct_function(sorted_graph, out_names)

            output = DiscretizedOperator(
                discr_params = tracer.discretization_params,
                preprocess_func = p_func,
                fields = fields,
                global_preprocess = global_preprocess,
                tracer = tracer
            )

            return output

        return wrapper
    return decorator

def make_op(out_discr, static_argnums=[], name=None):
    r'''Transforms a function into an operation, which is a single operator,
    registered in the computational graph'''
    def decorator(fun):
        def wrapped_fun(*args, **kwargs):
            assert len(kwargs.keys()) == 0  # Only positional arguments
            # Get the tracer
            for u in args:
                if hasattr(u, 'tracer'):
                    tracer = u.tracer
                    break
                
            # Add field names to inputs
            constants = {}
            inputs = []
            for i, arg in enumerate(args):
                # register constants
                if i in static_argnums:
                    const_name = f"_c_{tracer.counter()}"
                    constants[const_name] = arg
                else:
                    inputs.append(arg.name)

            # Construct output field 
            if name is None:
                f_name = f"{fun.__name__}"
            else:
                f_name = name
            u_name = f"_{tracer.counter()}"
            outfield = TracedField(out_discr, {}, tracer, u_name)

            # Add operation to the computational graph
            fun.__name__ = f_name
            op = Operation(fun, inputs, constants, outfield)
            tracer.add_operation(op)
            return outfield
        return wrapped_fun
    return decorator
