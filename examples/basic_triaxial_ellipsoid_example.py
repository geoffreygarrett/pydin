"""
NOTE: This example requires GSL, and as such, it is currently not supported by Windows and masOS.
See https://github.com/geoffreygarrett/pydin/issues/1 for more information.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pydin.core.gravitation import TriAxialEllipsoid
import pydin.core.logging as pdlog


class ModelParams:
    """Class for managing model parameters."""

    def __init__(self, a, b, c, rho, limit, n, z):
        self.a = a
        self.b = b
        self.c = c
        self.rho = rho
        self.limit = limit
        self.n = n
        self.z = z


def initialize_gravity(params):
    """Initializes the gravitational model."""
    G = 6.67408 * 1e-11
    mu = 4.0 / 3.0 * np.pi * G * params.rho * params.a * params.b * params.c
    return TriAxialEllipsoid(params.a, params.b, params.c, mu)


def create_meshgrid(params):
    """Creates a meshgrid for the x, y, z coordinates."""
    x = np.linspace(-params.limit, params.limit, params.n)
    y = np.linspace(-params.limit, params.limit, params.n)
    return np.meshgrid(x, y, np.array([params.z]))


def calculate_potential(gravity, X, Y, Z):
    """Calculates the potential using the provided gravity model."""
    timer_name = "Gravitational potential calculation"
    pdlog.start_timer(timer_name)
    U = gravity.calculate_potentials(X, Y, Z)
    pdlog.stop_timer(timer_name)
    return U


def create_contour_plot(X, Y, U, params, filename='gravitational_potential.png'):
    """Creates a contour plot and saves it to a file."""
    plt.figure(figsize=(10, 10), dpi=300)
    plt.contourf(X[:, :, 0], Y[:, :, 0], U[:, :, 0])

    ellipse = Ellipse(
        xy=(0, 0), width=2 * params.a, height=2 * params.b, angle=0, edgecolor='k', fc='None', lw=2, ls='--')
    plt.gca().add_patch(ellipse)
    plt.gca().set_aspect('equal')

    plt.title('Gravitational potential on X-Y plane at Z={}'.format(params.z))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(filename, dpi=300)


def run_example():
    pdlog.set_level(pdlog.DEBUG)
    pdlog.info("Starting tri-axial ellipsoid example")

    # Initialize parameters
    params = ModelParams(a=300.0, b=200.0, c=100.0, rho=2.8 * 1000.0, limit=1000.0, n=1000, z=0.0)

    # Initialize gravitational model
    gravity = initialize_gravity(params)

    # Create meshgrid
    X, Y, Z = create_meshgrid(params)

    # Calculate potential
    U = calculate_potential(gravity, X, Y, Z)

    # Create contour plot
    create_contour_plot(X, Y, U, params)

    pdlog.info("Finished tri-axial ellipsoid example, goodbye!")


if __name__ == '__main__':
    run_example()
