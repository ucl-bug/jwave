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
power = Operator("power")
power_scalar = Operator("power_scalar")
        
class derivative(Operator):
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, u):
        return u.discrertization.derivative(u, self.axis)

class elementwise(Operator):
    def __init__(self, func: Callable):
        self.func = func
    
    def __call__(self, u):
        return u.discretization.elementwise(u, self.func)