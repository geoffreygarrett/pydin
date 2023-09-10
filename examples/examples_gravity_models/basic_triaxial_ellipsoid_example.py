"""
NOTE: This example requires GSL, and as such, it is currently not supported by Windows and masOS.
See https://github.com/geoffreygarrett/pydin/issues/1 for more information.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pydin.core.logging as pdlog
from matplotlib.patches import Ellipse
from pydin.core.gravitation import TriAxialEllipsoid

matplotlib.use("QtAgg")


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


def calculate_potential(gravity, grid):
    """Calculates the potential using the provided gravity model."""
    timer_name = "Gravitational potential calculation"
    pdlog.start_timer(timer_name)
    U = gravity.potential_grid(grid)
    pdlog.stop_timer(timer_name)
    return U


from matplotlib.colors import LogNorm


def create_potential_plot(X, Y, U, params, filename="gravitational_potential.png"):
    """Creates a contour plot and saves it to a file."""
    plt.figure(figsize=(12, 12), dpi=300)
    plt.contourf(X[:, :, 0], Y[:, :, 0], U[:, :, 0])

    ellipse = Ellipse(
        xy=(0, 0),
        width=2 * params.a,
        height=2 * params.b,
        angle=0,
        edgecolor="k",
        fc="None",
        lw=2,
        ls="--",
    )

    # plot log scale

    plt.gca().add_patch(ellipse)
    plt.gca().set_aspect("equal")

    # add colorbar
    plt.colorbar()

    plt.title("Gravitational potential on X-Y plane at Z={}".format(params.z))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    plt.savefig(filename, dpi=300)


def create_jacobi_plot(X, Y, U, params, filename="jacobi.png"):
    """Creates a contour plot and saves it to a file."""
    plt.figure(figsize=(12, 12), dpi=300)

    # Create the contour plot on a log scale
    log_norm = LogNorm(vmin=np.min(U), vmax=np.max(U))
    contour = plt.contourf(
        X[:, :, 0], Y[:, :, 0], U[:, :, 0], levels=100, norm=log_norm
    )

    ellipse = Ellipse(
        xy=(0, 0),
        width=2 * params.a,
        height=2 * params.b,
        angle=0,
        edgecolor="k",
        fc="None",
        lw=2,
        ls="--",
    )

    plt.gca().add_patch(ellipse)
    plt.gca().set_aspect("equal")

    # Add colorbar
    colorbar = plt.colorbar(contour)

    plt.title("Gravitational potential on X-Y plane at Z={}".format(params.z))
    plt.xlabel("X")
    plt.ylabel("Y")

    # Save the plot before showing
    plt.savefig(filename, dpi=300)

    plt.show()


def calculate_rotational_jacobi_integral(grid, omega, U):
    # Equation: 1/2 * v dot v - 1/2 (omega cross r) dot (omega cross r) - U
    grid = np.array(grid)[..., 0]
    print("grid shape: ", grid.shape)

    # Make sure omega is a numpy array with shape (1, 1, 1, 3) for broadcasting
    omega = np.array(omega).reshape(3, 1, 1)

    # Compute omega cross r (cross product)
    omega_cross_r = np.cross(omega, grid, axis=0)
    print("omega cross r shape: ", omega_cross_r.shape)

    # Compute the square of the magnitude of omega cross r and multiply by 1/2
    rotational_potential = 0.5 * np.einsum("ijk,ijk->jk", omega_cross_r, omega_cross_r)
    print("rotational potential shape: ", rotational_potential.shape)

    # Compute the square of the magnitude of v (assuming v = grid) and multiply by 1/2
    # kinetic_energy_term = 0.5 * np.sum(grid ** 2, axis=-1)
    # print("kinetic energy term shape: ", kinetic_energy_term.shape)

    # Compute the rotational Jacobi integral
    return rotational_potential + U


def run_example():
    pdlog.set_level(pdlog.DEBUG)
    pdlog.info("Starting tri-axial ellipsoid example")

    # Initialize parameters
    params = ModelParams(
        a=300.0, b=200.0, c=100.0, rho=2.8 * 1000.0, limit=2000.0, n=100, z=0.0
    )

    # Initialize gravitational model
    gravity = initialize_gravity(params)

    # Create meshgrid
    grid = create_meshgrid(params)

    # Extract meshgrid coordinates
    X, Y, _ = grid

    # Calculate potential
    U = calculate_potential(gravity, grid)

    # Rotational potential
    omega = np.array([0.0, 0.0, 0.00005])
    print("max U: ", np.max(U))
    print("min U: ", np.min(U))

    # Calculate rotational Jacobi integral
    C = calculate_rotational_jacobi_integral(grid, omega, U)
    print("max C: ", np.max(C))
    print("min C: ", np.min(C))

    create_potential_plot(X, Y, C, params)

    pdlog.info("Finished tri-axial ellipsoid example, goodbye!")


if __name__ == "__main__":
    run_example()
