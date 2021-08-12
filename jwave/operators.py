from typing import Callable


class Operator(object):
    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            for arg in args:
                if not isinstance(arg, float) and not isinstance(arg, int):
                    return getattr(arg.discretization, self.name)(*args, **kwargs)
        else:
            for arg in kwargs.values():
                if not isinstance(arg, float) and not isinstance(arg, int):
                    return getattr(arg.discretization, self.name)(*args, **kwargs)

        raise RuntimeError(f"Operator {self.name} not found")


add = Operator("add")
add_scalar = Operator("add_scalar")
div = Operator("div")
invert = Operator("invert")
mul = Operator("mul")
mul_scalar = Operator("mul_scalar")
div = Operator("div")
div_scalar = Operator("div_scalar")
power = Operator("power")
power_scalar = Operator("power_scalar")
reciprocal = Operator("reciprocal")

gradient = Operator("gradient")
nabla_dot = Operator("nabla_dot")
diag_jacobian = Operator("diag_jacobian")
sum_over_dims = Operator("sum_over_dims")
laplacian = Operator("laplacian")

class OperatorWithArgs(Operator):
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, u):
        return getattr(u.discretization, self.name)(u, *self.args, **self.kwargs)

class elementwise(Operator):
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, u):
        return u.discretization.elementwise(u, self.func)


class dirichlet(Operator):
    def __init__(self, bc_bvalue):
        self.bc_value = bc_bvalue

    def __call__(self, v):
        return self.u.discretization.dirichlet(self.u, v)
