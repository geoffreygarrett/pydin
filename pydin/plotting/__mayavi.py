import configparser
import os
from collections import namedtuple
from functools import wraps

from pydin import attempt_import

mlab = attempt_import('mayavi.mlab')

from . import requires

# export ETS_TOOLKIT=qt4
# export QT_API=pyqt5



# Define common parameters as a dictionary
COMMON_PLOT_PARAMS = {
    'dark_mode': True,
    'show': False,
    'save': False,
    'filename': None,
    'figure_size': (400, 400),
    'magnification': 2
}

# Define common parameters as a NamedTuple
PlotParams = namedtuple('PlotParams', [
    'dark_mode',
    'show',
    'save',
    'filename',
    'figure_size',
    'magnification',
    'parallel_projection',
    'plot_title'
])
COMMON_PLOT_PARAMS = PlotParams(
    dark_mode=True,
    show=False,
    save=False,
    filename=None,
    figure_size=(400, 400),
    magnification=2,
    parallel_projection=False,
    plot_title=None,
)

BG_BLACK = (0, 0, 0)
BG_WHITE = (1, 1, 1)
BG_GRAY = (0.5, 0.5, 0.5)
BG_DARK_GRAY = (36 / 255, 31 / 255, 49 / 255)


#

def check_plot_params(kwargs):
    """Check if provided kwargs are part of the common plot parameters"""
    for key in kwargs:
        if key not in COMMON_PLOT_PARAMS:
            pass
            # logging.warn(f"Unknown parameter '{key}' passed. Allowed parameters are {list(COMMON_PLOT_PARAMS.keys())}")
            # raise ValueError(
            #     f"Unknown parameter '{key}' passed. Allowed parameters are {list(COMMON_PLOT_PARAMS.keys())}")


DEFAULT_STYLE = {
    'bgcolor': BG_DARK_GRAY,
    'fgcolor': (1, 1, 1),
    'color': BG_DARK_GRAY,
    'opacity': 0.6
}


def get_color(dark_mode):
    """Function to decide color based on the mode"""
    return (0.5, 0.5, 0.5) if dark_mode else (1, 1, 1)


def parse_tuple(string):
    try:
        return tuple(map(float, string.split(',')))
    except ValueError:
        raise ValueError("Failed to parse tuple from string: '{}'".format(string))


def parse_float(string):
    try:
        return float(string)
    except ValueError:
        raise ValueError("Failed to parse float from string: '{}'".format(string))


def load_style(style_name):
    """Load a style by name from the config file. If the style does not exist in the config file,
    the default style is returned.

    Parameters
    ----------
    style_name: str
        The name of the style to load.

    Returns
    -------
    dict
        A dictionary containing the style parameters.
    """
    # Load default style
    style_params = DEFAULT_STYLE.copy()

    # Read config file
    config = configparser.ConfigParser()
    config.read(os.path.expanduser('~/.bor/pydin/plotting.ini'))

    # Define type parser for each key
    parsers = {'bgcolor': parse_tuple, 'fgcolor': parse_tuple, 'color': parse_tuple, 'opacity': parse_float}

    # If the style exists in the config file, override the default style
    if style_name in config.sections():
        for key, parser in parsers.items():
            if key in config[style_name]:
                style_params[key] = parser(config.get(style_name, key))

    return style_params

    return style_params


def mayavi_style(style='dark', plot_params=None):
    """Decorator for Mayavi plots to apply a specific style and optional additional parameters.

    Parameters
    ----------
    style: str, optional
        The style to apply to the plot. This style is translated to an equivalent Mayavi style using the StyleRegistry.
    plot_params: dict, optional
        Additional or overriding parameters to apply to the plot.
    """
    if plot_params is None:
        plot_params = {}

    def decorator(func):
        @wraps(func)
        @requires('mayavi.mlab', 'numpy')
        def wrapper(*args, **kwargs):
            # Check if provided kwargs are part of the common plot parameters
            check_plot_params(kwargs)

            # Merge common parameters and additional parameters, with additional parameters taking precedence
            merged_params = COMMON_PLOT_PARAMS._asdict()
            merged_params.update(plot_params)
            merged_params.update(kwargs)

            # Load style from config file or fall back to default
            style_params = load_style(style)

            # Override style parameters with any explicitly given parameters
            style_params.update(kwargs)

            # Add the style params to kwargs
            kwargs.update(style_params)

            # Call the plotting function
            return func(*args, **kwargs)

        return wrapper

    return decorator
