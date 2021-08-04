from jwave.geometry import Domain
from jwave import operators as jops
from typing import List, Union, Any
from collections.abc import Iterable
from functools import wraps
from hashids import Hashids


class Discretization(object):
    domain: Domain

    def init_params(self, seed):
        return 0

    def __repr__(self):
        return self.__class__.__name__


class Globals(object):
    def __init__(self):
        self.dict = {"shared": {}, "independent": {}}

    def get(self, group=None, key=None):
        if group is None and key is None:
            return self.dict
        return self.dict[group][key]

    def set(self, key: str, value: Any, group: str):
        if group == "shared":
            if key not in self.dict["shared"].keys():
                self.dict["shared"][key] = value
            else:
                print(f"{key} already exists in shared Globals, skipping")
        elif group == "independent":
            if key not in self.dict["independent"].keys():
                self.dict["independent"][key] = value
            else:
                raise ValueError(f"{key} already exists in independent Globals!")

    def __repr__(self):
        return f"{self.dict}"


class Tracer(object):
    def __init__(self):
        self.input_fields = {}
        self.globals = Globals()
        self.operations = {}
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

    def construct_function(self, sorted_graph: dict, names: List[str]):
        concrete_ops = {}

        def f(global_params, field_params):
            # Add used inputs
            for field_name in field_params.keys():
                if field_name in sorted_graph:
                    concrete_ops[field_name] = field_params[field_name]

            # Compose operations present in the sorted grap
            for op_name in self.operations.keys():
                if op_name in sorted_graph:
                    op = self.operations[op_name]
                    args = [concrete_ops[n] for n in op.inputs]
                    if op.param_kind != "none":
                        op_params = global_params[op.param_kind][op.params]
                    else:
                        op_params = {}
                    concrete_ops[op_name] = op.fun(op_params, *args)

            # Output requested parameters
            outputs = []
            for n in names:
                outputs.append(concrete_ops[n])
            return outputs

        return f

    @staticmethod
    def _filter_unneded_ops(operations: dict, out_fields: List[str]):
        op_keys = [k for k in operations.keys()]
        op_args = [k.inputs for k in operations.values()]
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
        ops = ["- " + str(x) + "\n" for x in self.operations.values()]
        ops = "".join(ops)
        input_keys = tuple([*self.input_fields.keys()])
        string = f"Input fields: {input_keys}\n\n"
        string += f"Globals: {self.globals}\n\n"
        string += f"Operations:\n{ops}"
        return string


def is_numeric(x):
    return (type(x) == int)+(type(x) == float)+(type(x) == complex)


class Field(object):
    def __init__(self, discretization, name, params):
        self.params = params
        self.discretization = discretization
        self.name = name

    def __call__(self, x):
        return self.discretization.get_field()(self.params, x)

    def get_field(self):
        return self.discretization.get_field()

    def init_params(self, seed):
        return self.discretization.init_params(seed)

    def __add__(self, other):
        if is_numeric(other):
            return jops.add_scalar(self, other)
        else:
            return jops.add(self, other)

    def __truediv__(self, other):
        if is_numeric(other):
            return jops.div_scalar(self, other)
        else:
            return jops.div(self, other)

    def __mul__(self, other):
        if is_numeric(other):
            return jops.mul_scalar(self, other)
        else:
            return jops.mul(self, other)

    def __pow__(self, other):
        if is_numeric(other):
            return jops.power_scalar(self, other)
        else:
            return jops.power(self, other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        self = jops.reciprocal(self)
        return self*other

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
        self, discretization: Discretization, params: dict, tracer: Tracer, name: str
    ):
        self.params = params
        self.discretization = discretization
        self.tracer = tracer
        self.name = name
        self.preprocess_params = lambda x, y: y  # Identity function for parameters

    @staticmethod
    def from_field(field, tracer, name):
        return TracedField(field.discretization, field.params, tracer, name)

    def __repr__(self):
        return f"-- TracedField :: {self.discretization}, {self.name} \n\n{self.tracer}"

    def get_field(self):
        f = self.discretization.get_field()

        def wrapped_f(global_params, input_params, x):
            new_params = self.preprocess_params(global_params, input_params)
            return f(new_params, x)

        wrapped_f.__name__ = f.__name__
        return wrapped_f

    def get_field_on_grid(self):
        f = self.discretization.get_field_on_grid()

        def wrapped_f(global_params, input_params):
            new_params = self.preprocess_params(global_params, input_params)
            return f(new_params)

        wrapped_f.__name__ = f.__name__
        return wrapped_f


class DiscretizedOperator(object):
    def __init__(
        self, global_params, preprocess_func, fields, global_preprocess, tracer
    ):
        self.globals = global_params
        self.preprocess_func = preprocess_func
        self.fields = fields
        self.global_preprocess = global_preprocess
        self.tracer = tracer

    def __repr__(self):
        discr = [x.discretization for x in self.fields]
        names = [x.name for x in self.fields]
        return f"DiscretizedOperator :: {discr}, {names} \n\n {self.tracer}"

    def get_global_params(self):
        return self.tracer.globals.get()

    def get_field(self, idx=-1):
        """idx=-1 means all fields"""
        if idx != -1:
            f = self.fields[idx].discretization.get_field()
            preprocess = self.preprocess_func[idx]

            def wrapped_f(global_params, input_params, x):
                new_params = preprocess(global_params, input_params)
                return f(new_params, x)

            return wrapped_f
        else:
            preprocess = self.global_preprocess
            f_all = [f.discretization.get_field() for f in self.fields]

            def wrapped_f(global_params, input_params, x):
                all_new_params = preprocess(global_params, input_params)
                return [f(all_new_params[i], x) for i, f in enumerate(f_all)]

            return wrapped_f

    def get_field_on_grid(self, idx=-1):
        """idx=-1 means all fields"""
        if idx != -1:
            f = self.fields[idx].discretization.get_field_on_grid()
            preprocess = self.preprocess_func[idx]

            def wrapped_f(global_params, input_params):
                new_params = preprocess(global_params, input_params)
                return f(new_params)

            return wrapped_f
        else:
            preprocess = self.global_preprocess
            f_all = [f.discretization.get_field_on_grid() for f in self.fields]

            def wrapped_f(global_params, input_params):
                all_new_params = preprocess(global_params, input_params)
                return [
                    f(all_new_params[i]) for i, f in enumerate(f_all)
                ]

            return wrapped_f


def operator(has_aux=False, debug=False):
    """Returns a decorator that builds the computational graph of an operator"""
    if has_aux == True:
        raise NotImplementedError(
            "Operator with auxiliary arguments are not currently supported."
        )

    def decorator(fun):
        def wrapper(*args, **kwargs):

            tracer = Tracer()

            # Use only named variables
            if len(args) > 0:
                raise ValueError(
                    "Only keyword arguments can be used in a traced function"
                )

            # Register input fields
            for input_name in kwargs.keys():
                field = TracedField.from_field(
                    kwargs[input_name], tracer, name=input_name
                )
                tracer.add_input_field(name=input_name, field=field)

            # Register operations
            out_fields = fun(*[x for x in tracer.input_fields.values()])
            if not isinstance(out_fields, Iterable):
                out_fields = (out_fields,)

            # Register discretization parameters
            """
            for f in out_fields:
                tracer.add_discretization(f.discretization)
            """

            # Generates jax function for each output variable
            out_names = [x.name for x in out_fields]
            p_func = []
            fields = []
            for out in out_names:
                sorted_graph = tracer._filter_unneded_ops(tracer.operations, [out])
                if debug:
                    print(f"Sorted graph for {out}:\n{sorted_graph}\n")
                pf = tracer.construct_function(sorted_graph, [out])
                preprocess_func = lambda x, y: pf(x, y)[0]  # Only need one output
                field = tracer.operations[out].yelds

                p_func.append(preprocess_func)
                fields.append(field)

            # Making function preprocessing parameters for all fields
            sorted_graph = tracer._filter_unneded_ops(tracer.operations, out_names)
            global_preprocess = tracer.construct_function(sorted_graph, out_names)

            output = DiscretizedOperator(
                global_params=tracer.globals,
                preprocess_func=p_func,
                fields=fields,
                global_preprocess=global_preprocess,
                tracer=tracer,
            )

            return output

        return wrapper

    return decorator
