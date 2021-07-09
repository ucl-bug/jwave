class AddScalar(object):
    def __init__(self, scalar):
        self.scalar = scalar

    def discrete_transform(self, seed, discr_params, field_params):
        return field_params
    
    def setup(self, seed, field):
        '''Returns the initialized parameters and the output
        discretization family'''
        def f(discr_params, field_params, x):
            scalar = field_params['scalar']
            return field.discr.get_field(discr_params, field_params, x) + scalar
        new_discretization = Arbitrary(field.domain, f, field.discr.init_params)

        op_parameters = {"scalar": self.scalar}

        return op_parameters, new_discretization

class AddScalarLinear(object):
    def __init__(self, scalar, tunable_parameters=False):
        self.scalar = scalar

    def discrete_transform(self, seed, discr_params, field_params):
        return field_params + discr_params["scalar"]
    
    def setup(self, seed, field):
        '''Same discretization family as the input'''
        new_discretization = field.discretization
        parameters = {"scalar": self.scalar}
        return parameters, new_discretization

