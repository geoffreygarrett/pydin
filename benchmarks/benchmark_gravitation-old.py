import numpy as np
from pydin.core.gravitation import TriAxialEllipsoid

from utils import run_benchmarks

# Define parameters
a, b, c = 300.0, 200.0, 100.0
rho = 2.8 * 1000.0
G = 6.67408 * 1e-11
mu = 4.0 / 3.0 * np.pi * G * rho * a * b * c


# Define a function to vectorize 3D inputs
def vectorize_3d_input(func):
    def wrapped_func(x, y, z):
        return func(np.array([x, y, z]))

    return wrapped_func


# Create the ellipsoid
ellipsoid = TriAxialEllipsoid(a, b, c, mu)

# Vectorize the potential method
vectorized_potential = np.vectorize(vectorize_3d_input(ellipsoid.potential))

grav_methods = {
    ellipsoid.potential: [{}],  # The parameters are already defined in the instance
    ellipsoid.potential_series: [{}],
    ellipsoid.potential_grid: [{}],
    vectorized_potential: [{}],
}

custom_plot_settings = {
    'fill_between_color': 'lightblue',
    'fill_between_alpha': 0.3,
    'linestyles': ['-', ':', '--', '-.', (0, (1, 10))],
    'xlabel': 'Grid Size (log scale)',
    'ylabel': 'Execution Time (s, log scale)',
    'title': 'Performance Benchmarking',
    'figure_size': (15, 10),
}

run_benchmarks(func_dict=grav_methods,
               single_input=False,  # The input is a 3D grid
               save_as='benchmark_single_input.png',
               show=False, end=6,
               num_points=20,
               use_error_bars=True, **custom_plot_settings)
