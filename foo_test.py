from pydin.core.gravitation import TriAxialEllipsoid
import pydin
import pydin.core.logging as pdlog
import numpy as np
# Plot the results
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def test_tri_axial_ellipsoid():
    pdlog.set_level(pdlog.DEBUG)
    pdlog.info("Starting tri-axial ellipsoid example")

    # Parameters for the Tri-axial Ellipsoid gravitational model
    a = 300.0
    b = 200.0
    c = 100.0
    rho = 2.8 * 1000.0
    G = 6.67408 * 1e-11
    mu = 4.0 / 3.0 * np.pi * G * rho * a * b * c

    # Initialize the gravitational model
    gravity = TriAxialEllipsoid(a, b, c, mu)
    # gravity = HollowSphere(np.sqrt(a ** 2 + b ** 2 + c ** 2), mu)


    # Create grid of points
    limit = 1000.0
    n = 1000
    x = np.linspace(-limit, limit, n)
    y = np.linspace(-limit, limit, n)
    z = np.array([0.0])

    # Create the meshgrid
    X, Y, Z = np.meshgrid(x, y, z)

    # Compute the potential resulting from the gravitational model
    timer_name = "Gravitational potential calculation"
    pdlog.start_timer(timer_name)
    U = gravity.calculate_potentials(X, Y, Z)
    pdlog.stop_timer(timer_name)

    # # Compute the potential resulting from the gravitational model
    # potential_vectorized = np.vectorize(gravity.potential, signature='(n)->()')
    # pdlog.start_timer("Vectorized potential calculation")
    # U_vectorized = potential_vectorized(np.dstack([X, Y, Z]))
    # pdlog.stop_timer("Vectorized potential calculation")

    # Create the contour plot
    plt.figure(figsize=(10, 10), dpi=300)
    plt.contourf(X[:, :, 0], Y[:, :, 0], U[:, :, 0])

    # Plot ellipsoid patch
    ellipse = Ellipse(xy=(0, 0), width=2 * a, height=2 * b, angle=0, edgecolor='k', fc='None', lw=2, ls='--')
    plt.gca().add_patch(ellipse)

    # Set the aspect ratio to 'equal'
    plt.gca().set_aspect('equal')

    # Customize the plot
    plt.title('Gravitational potential on X-Y plane at Z={}'.format(0.0))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('gravitational_potential.png', dpi=300)

    # Goodbye :)
    pdlog.info("Finished tri-axial ellipsoid example, goodbye!")


from matplotlib.animation import FuncAnimation

# Initialize contour levels with None
contour_levels = None


# Update function for the animation
# Update function for the animation
def update_fig(i, a_values, b_values, c_values, mu_values, X, Y, Z, ax):
    global contour_levels  # Use the global contour_levels variable

    ax.clear()
    a = a_values[i]
    b = b_values[i]
    c = c_values[i]
    mu = mu_values[i]

    gravity = TriAxialEllipsoid(a_values[i], b_values[i], c_values[i], mu_values[i])
    potential_vectorized = np.vectorize(gravity.potential, signature='(n)->()')
    U = potential_vectorized(np.dstack([X, Y, Z]))
    del gravity
    # U = gravity.calculate_potentials(X, Y, Z)
    # del gravity

    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Compute contour levels if it's the first frame
    if contour_levels is None:
        U_range = np.max(U) - np.min(U)
        contour_levels = np.linspace(np.min(U) - 0.5 * U_range, np.max(U) + 0.3 * U_range, 20)  # For example, 20 levels

    # Use stored contour levels
    ax.contourf(X[:, :, 0], Y[:, :, 0], U[:, :], levels=contour_levels)

    # ensure a > b
    if a < b:
        a, b = b, a

    ellipse = Ellipse(xy=(0, 0), width=2 * a, height=2 * b, angle=0, edgecolor='k', fc='None', lw=2, ls='--')
    ax.add_patch(ellipse)


# Tri axial ellipsoid animation function
def tri_axial_ellipsoid_animation():
    rho = 2.8 * 1000.0
    G = 6.67408 * 1e-11
    limit = 1000.0
    n = 200
    x = np.linspace(-limit, limit, n)
    y = np.linspace(-limit, limit, n)
    z = np.array([0.0])
    X, Y, Z = np.meshgrid(x, y, z)
    a_values = np.linspace(300.0, 100.0, 100)
    b_values = np.linspace(100.0, 300.0, 100)
    c_values = np.ones_like(a_values) * 100.0
    mu_values = 4.0 / 3.0 * np.pi * G * rho * a_values * b_values * c_values

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ani = FuncAnimation(fig, update_fig, frames=len(a_values),
                        fargs=(a_values, b_values, c_values, mu_values, X, Y, Z, ax), blit=False, interval=50)
    ani.save('gravitational_potential.gif', writer='pillow')
    plt.close(fig)


if __name__ == '__main__':
    tri_axial_ellipsoid_animation()
