"""plotting/__init__.py
plotting
========

The `plotting` module provides a collection of functions to create 2D and 3D plots using different libraries,
such as matplotlib, mayavi, and potentially more.

It implements an elegant and maintainable structure for library-specific submodules and a unified way of handling
the common scenario where certain libraries might not be installed.

The module includes:

- A utility to attempt importing a given module and handling `ImportError` gracefully.
- A decorator `requires` to specify the dependencies of a function, that will raise an `ImportError` if the required
  packages are not installed.
- A `StyleRegistry` class to map equivalent styles across different plotting libraries.
- A `plot_style` context manager for applying these styles in a pythonic manner.

Modules
-------
matplotlib : submodule for matplotlib specific plotting functions
mayavi     : submodule for mayavi specific plotting functions

"""

import importlib
import sys
from functools import wraps

# import logging as std_logging
from pydin import logging

logger = logging.stdout_color_mt(__name__)


def check_and_fix_qt():
    try:
        PySide6 = importlib.import_module('PySide6')
        QtCore = importlib.import_module('PySide6.QtCore')

        # Check if MidButton attribute exists, if not, assign it
        if not hasattr(QtCore.Qt, 'MidButton'):
            QtCore.Qt.MidButton = QtCore.Qt.MiddleButton
            logger.info("PySide6 MidButton alias assigned as MiddleButton")
    except ImportError:
        pass
    except Exception as e:
        print("An unexpected error occurred: ", e)
        sys.exit(1)


# fix Qt4/Qt6 pyside6 compatibility issues
check_and_fix_qt()


def attempt_import(module_name):
    """ Attempt to import a module. """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


# # Attempt to import the pydin.core.logging module, default to standard logging if not found
# logging = attempt_import('pydin.core.logging') or std_logging


def attempt_import(module_name):
    """
    Attempt to import a module.

    Parameters
    ----------
    module_name : str
        The name of the module to import.

    Returns
    -------
    module or None
        The imported module or None if the module cannot be imported.

    Examples
    --------
    >>> matplotlib = attempt_import('matplotlib')
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def fallback(message):
    """
    Provide a fallback function when the library is not available.

    Parameters
    ----------
    message : str
        The error message to display when the fallback function is called.

    Returns
    -------
    function
        A fallback function that raises an ImportError with the specified message when called.

    Examples
    --------
    >>> @fallback("Function requires matplotlib.")
    >>> def plot():
    >>>     pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            raise ImportError(message)

        return wrapper

    return decorator


def requires(*packages, severity='error'):
    """
    Decorator to check if all required packages are installed.

    Parameters
    ----------
    *packages : str
        One or more names of required packages.
    severity : {'error', 'warning'}, default 'error'
        The severity level of the action when a package is missing. If 'error', an ImportError is raised. If 'warning',
        a warning message is printed.

    Returns
    -------
    function
        The original function, wrapped to include a package requirement check.

    Examples
    --------
    >>> @requires('matplotlib')
    >>> def plot():
    >>>     pass
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            for package in packages:
                if not attempt_import(package):
                    msg = f"Missing required package '{package}'. Please install it to use '{func.__name__}' function."
                    if severity == 'warning':
                        logging.warning(msg)
                    else:
                        raise ImportError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


class StyleRegistry:
    """
    Registry for equivalent styles across different libraries.

    Examples
    --------
    >>> StyleRegistry.translate('matplotlib', 'dark')
    >>> StyleRegistry.add_style('mylib', 'mystyle', 'mycustomstyle')
    """

    _STYLES = {
        'matplotlib': {
            'dark': 'dark_background',
            'light': 'default',
            'classic': 'classic',
        },
        'mayavi': {
            'dark': 'black_bg',
            'light': 'white_bg',
            'classic': 'blue_red',
        },
        'plotly': {
            'dark': 'plotly_dark',
            'light': 'plotly_white',
            'classic': 'plotly',
        }
    }

    @classmethod
    def translate(cls, library, style):
        """
        Translate a style to a specific library.

        Parameters
        ----------
        library : str
            The name of the library.
        style : str
            The name of the style.

        Returns
        -------
        str
            The style name in the given library.

        Examples
        --------
        >>> StyleRegistry.translate('matplotlib', 'dark')
        """
        return cls._STYLES.get(library, {}).get(style, style)

    @classmethod
    def add_style(cls, library, style_name, style):
        """
        Add a new style for a specific library.

        Parameters
        ----------
        library : str
            The name of the library.
        style_name : str
            The name of the new style.
        style : str
            The style definition in the given library.

        Examples
        --------
        >>> StyleRegistry.add_style('mylib', 'mystyle', 'mycustomstyle')
        """
        cls._STYLES[library][style_name] = style


class plot_style:
    """
    Context manager for plot styling.

    Examples
    --------
    >>> with plot_style('dark'):
    >>>     plot()
    """

    def __init__(self, style):
        """
        Initialize the context manager.

        Parameters
        ----------
        style : str
            The name of the style to apply.
        """
        self.style = style
        self.matplotlib = attempt_import('matplotlib')
        self.mayavi = attempt_import('mayavi.mlab')

    def __enter__(self):
        """
        Enter the context manager, applying the specified style.
        """
        if self.matplotlib is not None:
            self.matplotlib_style = self.matplotlib.rcParams.copy()
            self.matplotlib.style.use(StyleRegistry.translate('matplotlib', self.style))

        if self.mayavi is not None:
            self.mayavi_style = self.mayavi.get_engine().current_style
            self.mayavi.get_engine().current_style = StyleRegistry.translate('mayavi', self.style)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager, restoring the original style.
        """
        if self.matplotlib is not None:
            self.matplotlib.rcParams.update(self.matplotlib_style)

        if self.mayavi is not None:
            self.mayavi.get_engine().current_style = self.mayavi_style

        if exc_type is not None:
            logging.error(f"Exception occurred: {exc_val}")
