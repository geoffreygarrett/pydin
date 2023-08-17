"""

This file handles initialization of the pydin package, including importing
submodules from pydin.core directly into the pydin namespace.

"""

import importlib
# import logging as std_logging
import sys
import types

import pydin.core.logging as logging

logger = logging.stdout_color_mt(__name__)


# wrap std logger so that getLogger becomes get_logger
# std_logging.get_logger = std_logging.getLogger


def attempt_import(module_name, package=None):
    """ Attempt to import a module and return a warning if failed. """

    # if logging is None:
    #     from pydin import logging
    try:
        return importlib.import_module(module_name, package)
    except ImportError:
        logger = logging.get_logger(__name__)
        logger.warn(f"Failed to import {module_name}")
        return None


# if logging := attempt_import('.core.logging', 'pydin'):
#     logger = logging.stdout_color_mt(__name__)
#
#
#     def create_logger(name):
#         global logger
#         logger.debug(f"Creating logger {name}")
#
#         _logger = logging.stdout_color_mt(name)
#         _logger.info(f"Created logger {name}")
#         return _logger
#
#
# else:
#     import logging as std_logging
#
#     logging = std_logging
#     logging.get_logger = std_logging.getLogger
#     logger = logging.getLogger(__name__)
#
#
#     def create_logger(name):
#         global logger
#         logger.debug(f"Creating logger {name}")
#
#         _logger = logging.getLogger(name)
#         _logger.info(f"Created logger {name}")
#         return _logger

# attempt_import.logger = logger


def import_submodules_into_parent(package_name, parent_name):
    """
    Recursively import all submodules under package_name and assign them into parent_name module.

    Parameters
    ----------
    package_name : str
        The name of the package from which to import submodules.
    parent_name : str
        The name of the package into which to import the submodules.

    """
    # Import the parent package explicitly
    parent_module = attempt_import(package_name)
    if parent_module is None:
        raise ImportError(f"Failed to import package {package_name}")

    for attr_name in dir(parent_module):
        attr = getattr(parent_module, attr_name)
        if isinstance(attr, types.ModuleType):
            setattr(sys.modules[parent_name], attr_name, attr)


#

def initialize():
    # logger = logging.get_logger(__name__) if logging.get_logger(__name__) is not None else logging.getLogger(__name__)
    # logger = logging.getLogger(__name__)
    # logger.handlers = []
    try:
        logger.info("Initializing pydin")
        import_submodules_into_parent('pydin.core', 'pydin')
        logger.info("Successfully initialized pydin")
    except Exception as e:
        logging.error(f"Failed to initialize pydin: {str(e)}")
        raise


initialize()
# from .core import jit
# from .core import logging
# from .core import tbb
# from .core import gravitation

# from .core import *
