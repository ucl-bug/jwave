def is_numeric(x):
    """
    Check if x is a numeric value, including complex.
    """
    return isinstance(x, (int, float, complex))