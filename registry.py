import logging
from functools import wraps

logger = logging.getLogger(__name__)
activities = {}
operations = {}  # Op registry
lossfunctions = {}
operations_id_overrides = {}  # How the op affects the directory name
operations_optargs = {}  # Parameters for the ops


class PluginRegistry:
    def __init__(self, plugins_file='plugins.conf'):
        """
        Class that imports modules based on newline-seperated list of module names.
        :param plugins_file:
        """
        self.plugins_file = plugins_file
        self.loaded = False

    def load(self):
        if not self.loaded:
            plugin_list = []
            # Read plugins
            try:
                with open(self.plugins_file, 'r') as f:
                    plugin_list = [line.rstrip() for line in f]
            except FileNotFoundError:
                pass

            # Add default plugins
            plugin_list.append('ops')
            plugin_list.append('activities')
            plugin_list.append('lossfunctions')

            # Load plugins
            for plugin_name in plugin_list:
                if not plugin_name:
                    continue
                try:
                    exec('from %s import *' % plugin_name)
                    logger.info("Loaded plugins '%s'" % plugin_name)
                except Exception as e:
                    logger.error("Failed to load plugin '%s': %s" % (plugin_name, str(e)))

            self.loaded = True


def op(name, optargs=None, id_override=None):
    """
    Defines the @op decorator
    """
    def decorator(func):
        operations[name] = func

        if optargs is not None:
            operations_optargs[name] = optargs

        if id_override is not None:
            operations_id_overrides[name] = id_override

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


def lossfunction(name):
    """
    Defines the @lossfunction decorator
    """
    def decorator(func):
        lossfunctions[name] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    return decorator


# Register all activities and ops
plugins = PluginRegistry()
plugins.load()
