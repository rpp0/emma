from functools import wraps

activities = {}
operations = {}  # Op registry
operations_optargs = {}


def op(name, optargs=None):
    """
    Defines the @op decorator
    """
    def decorator(func):
        operations[name] = func
        if optargs is not None:
            operations_optargs[name] = optargs

        @wraps(func)  # Copy function metadata to wrapper()
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def activity(name):
    """
    Defines the @activity decorator
    """
    def decorator(func):
        activities[name] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    return decorator


# Register all activities and ops
from activities import *
from ops import *
