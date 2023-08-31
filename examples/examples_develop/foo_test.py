"""
NOTE: This example requires GSL, and as such, it is currently not supported by Windows and masOS.
 See https://github.com/geoffreygarrett/pydin/issues/1 for more information.
"""

# setattr(PySide6.QtCore.Qt, 'MidButton', PySide6.QtCore.Qt.MiddleButton)

# os.environ['ETS_TOOLKIT'] = 'qt6'
# os.environ['QT_API'] = 'pyside6'
# os.environ['QT_QPA_PLATFORM'] = 'linuxfb'
# os.environ['QT_DEBUG_PLUGINS'] = '1'

import numpy as np
# print current directory
# set pydin.core as pydin
from pydin.core.gravitation import Polyhedral
from pydin.core.gravitation import TriAxialEllipsoid

import pydin

print(dir(pydin.core))
print(dir(pydin))
# from pydin.gravitation import TriAxialEllipsoid

# Bazel sets this environment variable
# print(os.environ)
# runfiles_dir = os.environ.get('RUNFILES_DIR')
# eros_50k_path = os.path.join(runfiles_dir, 'pydin/file/eros_50k.ply')
# eros_faces_path = os.path.join(runfiles_dir, 'pydin/Eros.face')
# eros_nodes_path = os.path.join(runfiles_dir, 'pydin/Eros.node')
#
# # check if the file exists
# if not os.path.isfile(eros_50k_path):
#     raise FileNotFoundError(f"File {eros_50k_path} not found")

# Define parameters
a, b, c = 300.0, 200.0, 100.0
rho = 2.8 * 1000.0
G = 6.67408 * 1e-11
mu = 4.0 / 3.0 * np.pi * G * rho * a * b * c

from pydin import logging

pdlog = logging.get_logger(__file__)


def test_potential_series():
    ellipsoid = TriAxialEllipsoid(a, b, c, mu)
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
                          ]) * 1000.0
    pdlog.info(f"Positions: {positions}")
    potentials = ellipsoid.potential_series(positions)
    pdlog.info(f"Potentials: {potentials}")
    assert potentials.shape[0] == positions.shape[0]


def test_acceleration_series():
    ellipsoid = TriAxialEllipsoid(a, b, c, mu)
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
                          ]) * 1000.0
    pdlog.info(f"Positions: {positions}")
    accelerations = ellipsoid.acceleration_series(positions)
    pdlog.info(f"Accelerations: {accelerations}")
    assert accelerations.shape == positions.shape


def test_potential_grid():
    ellipsoid = TriAxialEllipsoid(a, b, c, mu)

    limit = 1000.0
    n_points = 1000

    x = np.linspace(-limit, limit, n_points)
    y = np.linspace(-limit, limit, n_points)

    positions = np.meshgrid(x, y, [0.0], indexing='ij')

    potentials = ellipsoid.potential_grid(positions)

    # Create a contour plot of the potentials at Z=0 (XY plane)
    plt.contourf(positions[0][:, :, 0], positions[1][:, :, 0], potentials[:, :, 0])
    plt.colorbar()
    plt.title('Contour plot of potential in XY plane at Z=0')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('potential_tensor.png', dpi=300)


def test_acceleration_grid():
    ellipsoid = TriAxialEllipsoid(a, b, c, mu)

    limit = 700.0
    n_points = 50

    x = np.linspace(-limit, limit, n_points)
    y = np.linspace(-limit, limit, n_points)

    xv, yv, zv = np.meshgrid(x, y, [0.0], indexing='ij')

    av_x, av_y, av_z = ellipsoid.acceleration_grid((xv, yv, zv))

    # Calculate the magnitude of acceleration
    mag = np.sqrt(av_x ** 2 + av_y ** 2 + av_z ** 2)

    # Set the figure size and create a new figure
    fig, ax = plt.subplots(figsize=(15, 15), dpi=100)

    # Create a quiver plot of the acceleration vectors at Z=0 (XY plane)
    quiver = ax.quiver(xv[:, :, 0],
                       yv[:, :, 0],
                       av_x[:, :, 0],
                       av_y[:, :, 0],
                       mag[:, :, 0],
                       cmap='jet',
                       scale_units='xy',
                       angles='xy',
                       width=0.003)

    # Add colorbar
    cbar = plt.colorbar(quiver, ax=ax, format=LogFormatterMathtext())
    cbar.set_label('Acceleration magnitude', fontsize=15)

    # Set equal aspect ratio
    ax.set_aspect('equal')

    ax.set_title('Quiver plot of acceleration in XY plane at Z=0', fontsize=20)
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    # tight
    plt.tight_layout()

    # Saving the figure to a file
    plt.savefig('acceleration_vectors.png', dpi=300)

    # Show the plot
    plt.show()


from matplotlib.ticker import LogFormatterMathtext


def test_acceleration_grid_3d():
    ellipsoid = TriAxialEllipsoid(a, b, c, mu)

    limit = 700.0
    n_points = 20  # Reduce the number of points for performance in 3D

    x = np.linspace(-limit, limit, n_points)
    y = np.linspace(-limit, limit, n_points)
    z = np.linspace(-limit, limit, n_points)  # Add z-axis values for 3D

    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')  # Include z in the meshgrid

    av_x, av_y, av_z = ellipsoid.acceleration_grid((xv, yv, zv))

    # Calculate the magnitude of acceleration
    mag = np.sqrt(av_x ** 2 + av_y ** 2 + av_z ** 2)

    # Set the figure size and create a new figure
    fig = plt.figure(figsize=(15, 15), dpi=100)
    ax = fig.add_subplot(111, projection='3d')  # Set up a 3D plot

    # color by magnitude

    col = mag

    col = np.log10(col)
    col = (col.ravel() - col.min()) / col.ptp()
    # log
    col = np.concatenate((col, np.repeat(col, 2)))
    col = plt.cm.jet(col)

    quiver = ax.quiver(xv,
                       yv,
                       zv,
                       av_x,
                       av_y,
                       av_z,
                       cmap='jet',
                       lw=2,
                       normalize=True,
                       length=50.0,
                       colors=col)
    # Adjust the length parameter to fit your data
    # quiver = ax.quiver(xv, yv, zv, av_x, av_y, av_z, cmap='jet', lw=1, length=100.0)

    # Add colorbar
    cbar = plt.colorbar(quiver, ax=ax, format=LogFormatterMathtext())
    cbar.set_label('Acceleration magnitude', fontsize=15)

    ax.set_title('Quiver plot of acceleration', fontsize=20)
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.set_zlabel('Z', fontsize=15)  # Add a z-axis label for 3D

    plt.tight_layout()

    # Saving the figure to a file
    plt.savefig('3D_acceleration_vectors.png', dpi=300)

    # Show the plot
    plt.show()


# def initialize_gravity(a, b, c, rho):
#     """Initializes the gravitational model."""
#     G = 6.67408 * 1e-11
#     mu = 4.0 / 3.0 * np.pi * G * rho * a * b * c
#     return TriAxialEllipsoid(a, b, c, mu)
#
#
# def create_meshgrid(limit, n, z=0.0):
#     """Creates a meshgrid for the x, y, z coordinates."""
#     x = np.linspace(-limit, limit, n)
#     y = np.linspace(-limit, limit, n)
#     return np.meshgrid(x, y, np.array([z]))
#
#
# def calculate_potential(gravity, X, Y, Z):
#     """Calculates the potential using the provided gravity model."""
#     timer_name = "Gravitational potential calculation"
#     pdlog.start_timer(timer_name)
#     U = gravity.calculate_potentials(X, Y, Z)
#     pdlog.stop_timer(timer_name)
#     return U
#
#
# def create_contour_plot(X, Y, U, a, b, filename='gravitational_potential.png'):
#     """Creates a contour plot and saves it to a file."""
#     plt.figure(figsize=(10, 10), dpi=300)
#     plt.contourf(X[:, :, 0], Y[:, :, 0], U[:, :, 0])
#
#     ellipse = Ellipse(xy=(0, 0), width=2 * a, height=2 * b, angle=0, edgecolor='k', fc='None', lw=2, ls='--')
#     plt.gca().add_patch(ellipse)
#     plt.gca().set_aspect('equal')
#
#     plt.title('Gravitational potential on X-Y plane at Z={}'.format(0.0))
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.savefig(filename, dpi=300)
#
#
# def run_example():
#     pdlog.set_level(pdlog.DEBUG)
#     pdlog.info("Starting tri-axial ellipsoid example")
#
#     # Define parameters
#     a, b, c = 300.0, 200.0, 100.0
#     rho = 2.8 * 1000.0
#
#     # Initialize gravitational model
#     gravity = initialize_gravity(a, b, c, rho)
#
#     # Create meshgrid
#     X, Y, Z = create_meshgrid(1000.0, 3000)
#
#     # Calculate potential
#     with tbb.TBBControl() as tbb_ctrl:
#         tbb_ctrl.max_allowed_parallelism = tbb.hardware_concurrency()
#         U = calculate_potential(gravity, X, Y, Z)
#
#     # Create contour plot
#     create_contour_plot(X, Y, U, a, b)
#
#     pdlog.info("Finished tri-axial ellipsoid example, goodbye!")


def animation_test():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from pydin.core.gravitation import TriAxialEllipsoid

    # Define initial parameters
    a_init, b_init, c = 300.0, 200.0, 100.0
    rho = 2.8 * 1000.0
    G = 6.67408 * 1e-11
    mu = 4.0 / 3.0 * np.pi * G * rho * a_init * b_init * c
    n_frames = 300

    # Define the spatial grid
    limit = 1000.0
    n_points = 100
    x = np.linspace(-limit, limit, n_points)
    y = np.linspace(-limit, limit, n_points)
    positions = np.meshgrid(x, y, [0.0], indexing='ij')

    # Create a figure and axis for the animation
    fig, ax = plt.subplots()

    # Set initial plot properties
    ax.set_title('Contour plot of potential in XY plane at Z=0')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # No initial contour plot
    contour = [None]

    def update(frame):
        # Update the axes 'a' and 'b'
        a = a_init + 100 * np.sin(2 * np.pi * frame / n_frames)
        b = b_init + 100 * np.sin(2 * np.pi * frame / n_frames)

        # Calculate the new tensor potential
        ellipsoid = TriAxialEllipsoid(a, b, c, mu)
        potentials = ellipsoid.potential_grid(positions)

        # Update the contour plot
        if contour[0] is not None:
            for coll in contour[0].collections:
                coll.remove()
        contour[0] = ax.contourf(positions[0][:, :, 0], positions[1][:, :, 0], potentials[:, :, 0])
        return contour[0].collections

    # Create the animation
    ani = FuncAnimation(fig, update, frames=n_frames, blit=False)

    # Save the animation
    writer = FFMpegWriter(fps=20, bitrate=1800)
    ani.save('potential_grid_animation.mp4', writer=writer)


########################################################################################################################
########################################################################################################################
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt


def test_polyhedral_plot():
    import numpy as np
    import matplotlib.pyplot as plt
    from pydin.core.gravitation import Polyhedral

    # laod nodes and faces from Eros.node and Eros.face
    nodes = np.loadtxt('Eros.node')
    faces = np.loadtxt('Eros.face', dtype=np.int64)

    # first col is index, so remove it
    nodes = nodes[:, 1:]
    faces = faces[:, 1:]

    # # Define nodes and faces for a cube
    # nodes = np.array([
    #     [-250, -250, -250],
    #     [250, -250, -250],
    #     [250, 250, -250],
    #     [-250, 250, -250],
    #     [-250, -250, 250],
    #     [250, -250, 250],
    #     [250, 250, 250],
    #     [-250, 250, 250]
    # ])
    #
    # # Create faces for the cube
    # faces = np.array([
    #     [0, 1, 2], [0, 2, 3],  # Bottom face
    #     [4, 5, 6], [4, 6, 7],  # Top face
    #     [0, 1, 5], [0, 5, 4],  # Front face
    #     [2, 3, 7], [2, 7, 6],  # Back face
    #     [0, 3, 7], [0, 7, 4],  # Left face
    #     [1, 2, 6], [1, 6, 5]  # Right face
    # ])

    # Define the spatial grid
    limit = 1.0
    n_points = 40
    x = np.linspace(-limit, limit, n_points)
    y = np.linspace(-limit, limit, n_points)
    positions = np.meshgrid(x, y, [0.0], indexing='ij')

    # Create a polyhedral object and calculate potential
    polyhedron = Polyhedral(nodes, faces, 2675.0)
    potentials = polyhedron.potential_grid(positions)

    # Create a figure for the plot
    fig, ax = plt.subplots()

    # Set plot properties
    ax.set_title('Contour plot of potential in XY plane at Z=0')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    # Generate the contour plot
    contour = ax.contourf(positions[0][:, :, 0], positions[1][:, :, 0], potentials[:, :, 0])

    # Show the plot
    plt.savefig('polyhedral_contour.png')
    pdlog.info("Finished polyhedral contour plot example, goodbye!")


# Define nodes and faces for a cube
def define_geometry(file_path='Eros.node', file_faces='Eros.face'):
    nodes = np.loadtxt(file_path)
    faces = np.loadtxt(file_faces, dtype=np.int64)

    # first col is index, so remove it
    nodes = nodes[:, 1:]
    faces = faces[:, 1:]

    return nodes, faces


# Define the spatial grid
def define_spatial_grid(limit=700.0, n_points=20):
    x = np.linspace(-limit, limit, n_points)
    y = np.linspace(-limit, limit, n_points)
    z = np.linspace(-limit, limit, n_points)
    return np.meshgrid(x, y, z, indexing='ij')


# Create a polyhedral object and calculate acceleration
def calculate_acceleration(xv, yv, zv, nodes, faces, density=2675.0):
    polyhedron = Polyhedral(nodes, faces, density)
    return polyhedron.acceleration_grid((xv, yv, zv))


def calculate_potential(xv, yv, zv, nodes, faces, density=2675.0):
    polyhedron = Polyhedral(nodes, faces, density)
    return polyhedron.potential_grid((xv, yv, zv))


def calculate_potential(xv, yv, zv, nodes, faces, density=2675.0):
    polyhedron = Polyhedral(nodes, faces, density)
    return polyhedron.potential_grid((xv, yv, zv))


def calculate_potential_2d(x, y, z, nodes, faces, density=2675.0):
    # Call your original calculate_potential function with the created 3D grid
    polyhedron = Polyhedral(nodes, faces, density)
    vv = polyhedron.potential_grid([xv, yv, zv])  # Pass list of 3D numpy arrays
    return vv


def define_spatial_grid_z0(limit=1.0, n_points=40):
    x = np.linspace(-limit, limit, n_points)
    y = np.linspace(-limit, limit, n_points)
    return np.meshgrid(x, y, [0.0], indexing='ij')


def define_2d_spatial_grid(limit=1.0, n_points=40):
    x = np.linspace(-limit, limit, n_points)
    y = np.linspace(-limit, limit, n_points)
    return np.meshgrid(x, y, indexing='ij')


def plot_figure_setup(title,
                      xv,
                      yv,
                      data,
                      cmap='jet',
                      dark_mode=True,
                      figsize=(10, 10),
                      fig=None,
                      ax=None,
                      gridlines=True):
    plt.style.use('dark_background' if dark_mode else 'default')
    cmap = 'inferno' if dark_mode else cmap

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel('$X$', fontsize=15)
    ax.set_ylabel('$Y$', fontsize=15)

    # Explicitly set major and minor ticks properties
    ax.tick_params(axis='both',
                   which='major',
                   labelsize=10,
                   color='white' if dark_mode else 'black')
    ax.tick_params(axis='both', which='minor', labelsize=8, color='white' if dark_mode else 'black')

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.xaxis.get_major_ticks()[0].label1.set_visible(True)
    ax.yaxis.get_major_ticks()[0].label1.set_visible(True)

    if gridlines:
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_aspect('equal')

    return fig, ax, cmap


from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def create_colorbar(fig, ax, mappable, label, log=False):
    axins = inset_axes(
        ax,
        width="5%",  # width = 5% of parent_bbox width
        height="100%",  # height : 100%
        loc='lower left',
        bbox_to_anchor=(1.07, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    # # Instantiate the formatter
    # formatter = ScalarFormatter()
    # formatter.set_scientific(True)  # Use scientific notation
    # formatter.set_useMathText(True)  # Use math text

    format = ticker.ScalarFormatter()
    format.set_scientific(True)
    format.set_powerlimits((-1, 1))  # this line forces scientific notation for all numbers
    format.set_useMathText(True)

    cbar = fig.colorbar(mappable, cax=axins, format=format)
    cbar.set_label(label, fontsize=15, labelpad=20)
    return cbar


def plot_quiver_2d(xv,
                   yv,
                   av_x,
                   av_y,
                   title=r'Quiver plot of acceleration at $Z=0$',
                   cmap='jet',
                   dark_mode=True,
                   show=False,
                   log=False,
                   save=True,
                   filename='2D_acceleration_vectors.png',
                   fig=None,
                   ax=None):
    mag = np.sqrt(av_x ** 2 + av_y ** 2)
    av_x_normalized = av_x / mag
    av_y_normalized = av_y / mag

    if log:
        mag = np.log10(mag)
        cbar_label = r'$\log_{10}(\|\mathbf{a}\|)$'
    else:
        cbar_label = r'$\|\mathbf{a}\|$'

    fig, ax, cmap = plot_figure_setup(title, xv, yv, mag, cmap, dark_mode, fig=fig, ax=ax)

    quiver = ax.quiver(xv[:, :, 0],
                       yv[:, :, 0],
                       -av_x_normalized[:, :, 0],
                       -av_y_normalized[:, :, 0],
                       mag[:, :, 0],
                       cmap=cmap,
                       lw=1,
                       pivot='middle',
                       scale=30.0)

    cbar = create_colorbar(fig, ax, quiver, cbar_label)

    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig, ax


def plot_contour_2d(xv,
                    yv,
                    potentials,
                    title=r'Contour plot of potential at $Z=0$',
                    cmap='viridis',
                    dark_mode=True,
                    show=False,
                    log=False,
                    save=True,
                    filename='2D_potential_contour.png',
                    fig=None,
                    ax=None):
    if log:
        potentials = np.log10(potentials)
        cbar_label = r'$\log_{10}(V)$'
    else:
        cbar_label = r'$V$'

    fig, ax, cmap = plot_figure_setup(title, xv, yv, potentials, cmap, dark_mode, fig=fig, ax=ax)

    contour = ax.contourf(xv[:, :, 0], yv[:, :, 0], potentials[:, :, 0], cmap=cmap)
    cbar = create_colorbar(fig, ax, contour, cbar_label)

    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig, ax


def poly2d_quiver(xv, yv, zv, nodes, faces, density=2570.0, dark_mode=True):
    vv = calculate_potential(xv, yv, zv, nodes, faces, density)
    fig, ax = plot_contour_2d(xv, yv, vv, dark_mode=dark_mode)


def poly2d_contour(xv, yv, zv, nodes, faces, density=2570.0, dark_mode=True):
    av_x, av_y, _ = calculate_acceleration(xv, yv, zv, nodes, faces, density)
    fig, ax = plot_quiver_2d(xv, yv, av_x, av_y, dark_mode=dark_mode)


# TEST: Define nodes and faces for a cube

def plot_3d_mesh(title,
                 nodes,
                 faces,
                 cmap='jet',
                 dark_mode=True,
                 figsize=(10, 10),
                 show=False,
                 save=True,
                 filename='3D_mesh.png'):
    plt.style.use('dark_background' if dark_mode else 'default')
    cmap = 'inferno' if dark_mode else cmap

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title(title, fontsize=20)
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.set_zlabel('Z', fontsize=15)

    ax.tick_params(axis='both',
                   which='major',
                   labelsize=10,
                   color='white' if dark_mode else 'black')
    ax.tick_params(axis='both', which='minor', labelsize=8, color='white' if dark_mode else 'black')

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.zaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.xaxis.get_major_ticks()[0].label1.set_visible(True)
    ax.yaxis.get_major_ticks()[0].label1.set_visible(True)
    ax.zaxis.get_major_ticks()[0].label1.set_visible(True)

    # Create a mesh
    poly3d = [nodes[faces[i]] for i in range(len(faces))]
    ax.add_collection3d(
        Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    # Calculate the limits
    all_nodes = nodes[faces.ravel()]
    ax.set_xlim(all_nodes[:, 0].min(), all_nodes[:, 0].max())
    ax.set_ylim(all_nodes[:, 1].min(), all_nodes[:, 1].max())
    ax.set_zlim(all_nodes[:, 2].min(), all_nodes[:, 2].max())

    ax.view_init(elev=20, azim=-35)  # adjust viewing angle

    ax.set_aspect('auto')

    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig, ax


def plot_3d_mesh_mayavi(nodes,
                        faces,
                        limit=None,
                        plane_origin=None,
                        dark_mode=True,
                        show=False,
                        save=False,
                        filename='3D_mesh.png',
                        figure=None,
                        azimuth=180,
                        color=(0.5, 0.5, 0.5),
                        elevation=90,
                        figure_size=(400, 400),
                        magnification=2,
                        opacity=0.6,
                        n_limit=1e6):
    # if n_limit is not None:
    #     nodes, faces = decimate_mesh(nodes, faces, 20)

    # Set the color based on the mode
    # color = (0.5, 0.5, 0.5) if dark_mode else (1, 1, 1)

    # If no figure is provided, create a new one
    if figure is None:
        # Create the mayavi figure
        figure = mlab.figure(bgcolor=color, size=figure_size)

    # Add the surface\
    # cyan vertices
    # red edges
    # Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
    mesh = mlab.triangular_mesh(nodes[:, 0],
                                nodes[:, 1],
                                nodes[:, 2],
                                faces,
                                color=color,
                                opacity=opacity,
                                figure=figure,
                                representation='surface',
                                line_width=0.4,
                                tube_sides=0.1)

    # Add cross section plane if plane_origin is specified
    if plane_origin is not None:
        cut_plane = mlab.pipeline.scalar_cut_plane(mesh, plane_orientation='z_axes', figure=figure)
        cut_plane.implicit_plane.origin = plane_origin

    # Set the view
    mlab.view(azimuth=azimuth, elevation=elevation)

    # Calculate and set the limits
    if limit is not None:
        x_lim_min, x_lim_max = -limit, limit
        y_lim_min, y_lim_max = -limit, limit
        z_lim_min, z_lim_max = -limit, limit
        mlab.points3d(np.array([x_lim_min, x_lim_max]),
                      np.array([y_lim_min, y_lim_max]),
                      np.array([z_lim_min, z_lim_max]),
                      mode='2dcross',
                      color=(0, 0, 0),
                      opacity=0,
                      figure=figure)

    # Save or display the figure
    if save:
        mlab.savefig(filename, magnification=magnification)
    if show:
        mlab.show()

    return figure


# def plot_3d_quiver_mayavi(xv,
#                           yv,
#                           av_x,
#                           av_y,
#                           av_z,
#                           cmap='jet',
#                           dark_mode=True,
#                           show=False,
#                           save=True,
#                           filename='3D_acceleration_vectors.png'):
#     # Set the color based on the mode
#     color = (0.5, 0.5, 0.5) if dark_mode else (1, 1, 1)
#
#     # Create the mayavi figure
#     mlab.figure(bgcolor=color, size=(400, 400))
#
#     # Calculate the magnitude of the acceleration vectors
#     mag = np.sqrt(av_x ** 2 + av_y ** 2 + av_z ** 2)
#
#     # Normalize the acceleration vectors
#     av_x_normalized = av_x / mag
#     av_y_normalized = av_y / mag
#     av_z_normalized = av_z / mag
#
#     # Add the 3D quiver plot
#     mlab.quiver3d(xv,
#                   yv,
#                   0,
#                   av_x_normalized,
#                   av_y_normalized,
#                   av_z_normalized,
#                   scalars=mag,
#                   colormap=cmap,
#                   mode='arrow',
#                   scale_factor=0.5)
#
#     # Set the view
#     mlab.view(azimuth=180, elevation=90)
#
#     # Save or display the figure
#     if save:
#         mlab.savefig(filename, magnification=2)
#     if show:
#         mlab.show()
#


def cross_section_plane(x, y, plane_z):
    """Generate a plane at the given z coordinates"""
    return np.full((x.shape[0], y.shape[0]), plane_z)


def plot_3d_mesh_cross_section(title,
                               nodes,
                               faces,
                               z_val=0.0,
                               cmap='jet',
                               dark_mode=True,
                               figsize=(10, 10),
                               show=False,
                               save=True,
                               filename='3D_mesh_cross_section.png',
                               limit=None):
    plt.style.use('dark_background' if dark_mode else 'default')
    cmap = 'inferno' if dark_mode else cmap

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title(title, fontsize=20)
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.set_zlabel('Z', fontsize=15)

    # Create a mesh
    poly3d = [nodes[faces[i]] for i in range(len(faces))]
    ax.add_collection3d(
        Poly3DCollection(poly3d, facecolors='cyan', linewidths=0.5, edgecolors='r', alpha=.25))
    all_nodes = nodes[faces.ravel()]

    # Calculate the limits
    if limit:
        x_lim_min = -limit
        x_lim_max = limit
        y_lim_min = -limit
        y_lim_max = limit
        z_lim_min = -limit
        z_lim_max = limit

    else:
        x_lim_min = all_nodes[:, 0].min()
        x_lim_max = all_nodes[:, 0].max()
        y_lim_min = all_nodes[:, 1].min()
        y_lim_max = all_nodes[:, 1].max()
        z_lim_min = all_nodes[:, 2].min()
        z_lim_max = all_nodes[:, 2].max()

    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(y_lim_min, y_lim_max)
    ax.set_zlim(z_lim_min, z_lim_max)

    # Cross-section plane
    plane_z = z_val  # choose the level of cross-section
    x = np.linspace(x_lim_min, x_lim_max, 10)
    y = np.linspace(y_lim_min, y_lim_max, 10)
    X, Y = np.meshgrid(x, y)
    Z = cross_section_plane(X, Y, plane_z)
    ax.plot_surface(X, Y, Z, color='magenta', alpha=0.2)

    ax.view_init(elev=20, azim=-35)  # adjust viewing angle

    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig, ax


def create_axes(figure, limit):
    # # Create an axes object
    # axes = mlab.orientation_axes()
    #
    # # Set the labels of the axes
    # # axes = Axes()
    # axes.axes.x_label, axes.axes.y_label, axes.axes.z_label = 'X', 'Y', 'Z'
    #
    # # Set the color of the axes
    # axes.axes.property.color = (1, 1, 1)  # RGB, (1, 1, 1) is white
    # axes.axes.axis_label_text_property.color = (1, 1, 1)  # RGB, (1, 1, 1) is white

    axes = mlab.axes(figure,
                     nb_labels=0,
                     x_axis_visibility=False,
                     y_axis_visibility=False,
                     z_axis_visibility=False,
                     ranges=[-limit, limit, -limit, limit, -limit, limit])

    # Set the labels of the axes
    axes.axes.x_label, axes.axes.y_label, axes.axes.z_label = '', '', ''

    # Set the color of the axes
    # axes.axes.property.color = (1, 1, 1)  # RGB, (1, 1, 1) is white
    # axes.axes.axis_label_text_property.color = (1, 1, 1)  # RGB, (1, 1, 1) is white
    # center the axes
    axes.axes.use_ranges = True
    axes.axes.ranges = [-limit, limit, -limit, limit, -limit, limit]
    # axes
    axes.axes.property.line_width = 0.2  # Thin lines

    return axes


# Usage:
# figure = mlab.figure('3D Contour', bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(1000, 800))
# axes = create_axes(figure, limit)


def plot_boundary_surface(figure, limit, color=(1, 1, 1), opacity=0.1):
    # Create a white square to limit the view
    x = np.linspace(-limit, limit, 10)
    y = np.linspace(-limit, limit, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mlab.mesh(X, Y, Z, color=color, opacity=opacity, figure=figure)


# def plot_boundary_limit(figure, limit, color=(1, 1, 1), opacity=0.15, offset=0.0):
#     limit = limit * 1.05
#     mlab.plot3d([limit, limit, -limit, -limit, limit], [limit, -limit, -limit, limit, limit],
#                 [offset, offset, offset, offset, offset],
#                 color=color,
#                 opacity=opacity,
#                 tube_radius=0.01,
#                 figure=figure)


def plot_3d_quiver_mayavi(xv,
                          yv,
                          zv,
                          avx,
                          avy,
                          avz,
                          figure=None,
                          scale_factor=0.1,
                          color=(0, 0, 1)):
    # If no figure is provided, create a new one
    if figure is None:
        # Create the mayavi figure
        figure = mlab.figure(size=(1000, 800))

    # Add the quiver plot
    mlab.quiver3d(xv, yv, zv, avx, avy, avz)

    return figure


from mayavi.sources.parametric_surface import ParametricSurface

from tvtk.api import tvtk


def load_texture(filename):
    # Read the image data.
    img = tvtk.JPEGReader(file_name=filename)
    img.update()
    # Create texture object.
    texture = tvtk.Texture(input_connection=img.output_port, interpolate=1)
    return texture


def create_ellipsoid(a, b, c, scene, wireframe=True, color=(0, 0, 1), opacity=0.7, texture=None):
    source = ParametricSurface()
    source.function = 'ellipsoid'  # Starting with a sphere because 'ellipsoid' is not a supported function
    surface = mlab.pipeline.surface(source, opacity=0.0)  # Creating the surface here and setting the opacity to 0.0

    acto1 = surface.actor
    actor = surface.actor.actor
    actor.property.opacity = 0.0
    actor.mapper.scalar_visibility = False
    actor.property.backface_culling = True
    actor.property.specular = 0.1
    actor.orientation = np.array([1, 0, 0]) * 360
    actor.origin = np.array([0, 0, 0])
    actor.position = np.array([0, 0, 0])
    actor.scale = np.array([a, b, c])
    actor.property.opacity = opacity  # Setting the opacity back to your desired value

    # add texture
    if texture is not None:
        acto1.enable_texture = True
        acto1.tcoord_generator_mode = 'sphere'
        texture_path = '/home/geoffrey/Desktop/download.jpeg'
        texture = load_texture(texture_path)
        acto1.texture = texture
    else:
        acto1.property.color = color
    actor.property.representation = 'wireframe' if wireframe else 'surface'

    return surface


# Plot ellipsoid with specific parameters in existing figure
def plot_ellipsoid_mayavi(a, b, c, figure=None, wireframe=False, color=(1, 0, 0), opacity=0.2, texture=None):
    if figure is None:
        figure = mlab.figure(size=(800, 800))

    create_ellipsoid(a, b, c, figure.scene, wireframe, color, opacity, texture)
    # mlab.draw(figure)


# def plot_ellipsoid_mayavi(a, b, c, figure=None, wireframe=True, color=(0, 0, 1), opacity=0.7):
#     engine = Engine()
#     engine.start()
#     scene = engine.new_scene()
#     scene.scene.disable_render = True
#
#     visual.set_viewer(scene)
#
#     # Create the ellipsoid
#     a = 0.26490647
#     b = 0.26490647
#     c = 0.92717265
#     ellipsoid_surface = create_ellipsoid(a, b, c, scene, engine, wireframe=wireframe, color=color, opacity=opacity)
#
#     # Create axes arrows
#     Arrow_From_A_to_B(0, 0, 0, a, 0, 0, np.array([a, 0.4, 0.4]))
#     Arrow_From_A_to_B(0, 0, 0, 0, b, 0, np.array([0.4, b, 0.4]))
#     Arrow_From_A_to_B(0, 0, 0, 0, 0, c, np.array([0.4, 0.4, c]))
#
#     scene.background = (1.0, 1.0, 1.0)
#     scene.scene.disable_render = False
#
#     mlab.show()

from mayavi import mlab


def get_frustum_coords(pos, target, up, hfov, aspect, near, far):
    forward = (target - pos) / np.linalg.norm(target - pos)
    right = np.cross(up, forward)
    up = np.cross(forward, right)

    fc = pos + forward * far
    nc = pos + forward * near

    hfar = 2 * np.tan(np.deg2rad(hfov) / 2) * far
    wfar = hfar * aspect
    hnear = 2 * np.tan(np.deg2rad(hfov) / 2) * near
    wnear = hnear * aspect

    ftl = fc + (up * hfar / 2) - (right * wfar / 2)
    ftr = fc + (up * hfar / 2) + (right * wfar / 2)
    fbl = fc - (up * hfar / 2) - (right * wfar / 2)
    fbr = fc - (up * hfar / 2) + (right * wfar / 2)

    ntl = nc + (up * hnear / 2) - (right * wnear / 2)
    ntr = nc + (up * hnear / 2) + (right * wnear / 2)
    nbl = nc - (up * hnear / 2) - (right * wnear / 2)
    nbr = nc - (up * hnear / 2) + (right * wnear / 2)

    return {
        "ftl": ftl,
        "ftr": ftr,
        "fbl": fbl,
        "fbr": fbr,
        "ntl": ntl,
        "ntr": ntr,
        "nbl": nbl,
        "nbr": nbr,
    }


def interp_bilinear(tl, tr, bl, br, u, v):
    top = (1 - u) * tl + u * tr
    bottom = (1 - u) * bl + u * br
    return (1 - v) * top + v * bottom


def get_pixel_directions(pos, target, up, hfov, aspect, near, far, v_resolution, h_resolution):
    frustum_coords = get_frustum_coords(pos, target, up, hfov, aspect, near, far)
    ftl, ftr, fbl, fbr = frustum_coords["ftl"], frustum_coords["ftr"], frustum_coords["fbl"], frustum_coords["fbr"]

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.linspace(0, 1, h_resolution), np.linspace(0, 1, v_resolution), indexing='ij')

    # Reshape u and v for broadcasting
    u = u[..., np.newaxis]
    v = v[..., np.newaxis]

    # Perform bilinear interpolation to find the pixel positions in 3D space
    pixel_positions = interp_bilinear(ftl, ftr, fbl, fbr, u, v)

    # Subtract the camera's position to get the directions
    directions = pixel_positions - pos

    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]

    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    z_flat = z.reshape(-1)

    # Normalize the directions
    directions = np.stack([x_flat, y_flat, z_flat], axis=1)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    return directions.T


def plot_frustum(pos, target, up, hfov, aspect, near, far, color=(0, 0, 1), opacity=1, tube_radius=None):
    frustum_coords = get_frustum_coords(pos, target, up, hfov, aspect, near, far)

    lines = [
        [frustum_coords["ntl"], frustum_coords["ntr"]],
        [frustum_coords["ntr"], frustum_coords["nbr"]],
        [frustum_coords["nbr"], frustum_coords["nbl"]],
        [frustum_coords["nbl"], frustum_coords["ntl"]],
        [frustum_coords["ftl"], frustum_coords["ftr"]],
        [frustum_coords["ftr"], frustum_coords["fbr"]],
        [frustum_coords["fbr"], frustum_coords["fbl"]],
        [frustum_coords["fbl"], frustum_coords["ftl"]],
        [frustum_coords["ntl"], frustum_coords["ftl"]],
        [frustum_coords["ntr"], frustum_coords["ftr"]],
        [frustum_coords["nbr"], frustum_coords["fbr"]],
        [frustum_coords["nbl"], frustum_coords["fbl"]],
    ]

    for line in lines:
        mlab.plot3d(*zip(*line), color=color, opacity=opacity, tube_radius=tube_radius)


# def calculate_plane_params(a, b, c, x, y, z):
#     # Calculate the plane parameters
#     # λ = l / k
#     # µ = m / k
#     # ν = n / k
#     # h = f / k
#     # k = sqrt(l ** 2 + m ** 2 + n ** 2)
#     # P = (x,y,z) = ( a ** 2 * l / f, b ** 2 * m / f, c ** 2 * n / f)
#     # l * x + m * y + n * z = f
#
#
#     return λ, µ, ν, h
def calculate_plane_params(a, b, c, p1, p2, p3):
    p1 = -1
    p2 = 1
    p3 = 1
    # Calculate plane parameters l, m, n
    l, m, n = - np.array([p1, p2, p3]) / np.linalg.norm(np.array([p1, p2, p3]))

    # Calculate the normalization factor k
    k = np.sqrt(l ** 2 + m ** 2 + n ** 2)

    f = 100

    # Calculate the plane parameters λ, µ, ν, h
    λ = l / k
    µ = m / k
    ν = n / k
    h = 1
    print(λ, µ, ν, h)

    return λ, µ, ν, h


from pydin.core import rot2, rot3


# def ellipse_of_intersection(A, B, C, D, a, b, c):


def normalize(val, ax, ay, az, c):
    """
    Helper function to normalize values.
    """
    print(f"ZeroDivisionError: {val}, {ax}, {ay}, {az}, {c}")

    try:
        return 1 / val ** 2 + (ax ** 2) / (az ** 2 * c ** 2)
    except ZeroDivisionError:
        print(f"ZeroDivisionError: {val}, {ax}, {ay}, {az}, {c}")
        return 0


# def ellipse_of_intersection(ax, ay, az, ad, a, b, c):
#     # Normalize ax + by + cz + d = 0
#     k = np.linalg.norm(np.array([ax, ay, az]))
#     l_1 = ax / k
#     m_1 = ay / k
#     n_1 = az / k
#     f_1 = ad / k
#
#     # # make n_1 the largest value
#     # if abs(n_1) < abs(m_1):
#     #     l_1, m_1, n_1 = l_1, n_1, m_1
#     #
#     # if abs(n_1) < abs(l_1):
#     #     l_1, m_1, n_1 = n_1, m_1, l_1
#
#     ax, ay, az, ad = l_1, m_1, n_1, f_1
#
#     normalized_a = n_1 ** 2 * c ** 2 / (a ** 2) + l_1 ** 2
#     normalized_b = 2 * l_1 * m_1
#     normalized_c = n_1 ** 2 * c ** 2 / (b ** 2) + m_1 ** 2
#     normalized_d = - 2 * l_1 * f_1
#     normalized_e = - 2 * m_1 * f_1
#     normalized_f = f_1 ** 2 - n_1 ** 2 * c ** 2
#
#     m0 = np.array([
#         [normalized_f, normalized_d / 2, normalized_e / 2],
#         [normalized_d / 2, normalized_a, normalized_b / 2],
#         [normalized_e / 2, normalized_b / 2, normalized_c]
#     ])
#
#     m = np.array([
#         [normalized_a, normalized_b / 2],
#         [normalized_b / 2, normalized_c]
#     ])
#
#     # Get eigenvalues and eigenvectors
#     eigvals, eigvecs = np.linalg.eig(m)
#
#     lambda1 = np.min(eigvals)
#     lambda2 = np.max(eigvals)
#
#     # Calculate the parameters a0 and b0
#     a0 = np.sqrt(- np.linalg.det(m0) / (np.linalg.det(m) * lambda1))
#     b0 = np.sqrt(- np.linalg.det(m0) / (np.linalg.det(m) * lambda2 / 3))
#
#     # Calculate the center of the ellipse
#     x0 = (normalized_b * normalized_e - 2 * normalized_c * normalized_d
#           ) / (
#                  4 * normalized_a * normalized_c - normalized_b ** 2
#          )
#     y0 = (normalized_b * normalized_d - 2 * normalized_a * normalized_e) / (
#             4 * normalized_a * normalized_c - normalized_b ** 2)
#     z0 = - (f_1 - m_1 * y0 - l_1 * x0) / n_1
#
#     print(f"Center of ellipse: {x0}, {y0}, {z0}")
#     # Calculate the angles theta, phi and omega
#
#     if normalized_a == normalized_c:
#         theta = np.pi / 4
#     else:
#         theta = np.arctan(normalized_b / (normalized_a - normalized_c)) / 2
#
#     qx = l_1 * ad / k
#     qy = m_1 * ad / k
#     qz = n_1 * ad / k
#     phi = np.pi / 2 - np.arctan2(qz, np.sqrt(qx ** 2 + qy ** 2))
#
#     if qx == 0:
#         omega = np.pi - np.pi / 2
#     else:
#         omega = np.pi - np.arctan2(qy, qx)
#
#     # Create transformation matrix
#     t = np.eye(4)
#     t[:3, :3] = rot3(-omega) @ rot2(-phi) @ rot3(2*theta)
#     t[:3, 3] = np.array([x0, y0, z0])
#
#     return (a0, b0), (x0, y0, z0), t, theta

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


# from sympy import symbols, expand, simplify
#
# # define the symbols
# X, Y, a, b, c, X_OT, Y_OT, Z_OT, psi_T, eps_T, omega_T = symbols('X Y a b c X_OT Y_OT Z_OT psi_T eps_T omega_T')
#
# # define additional symbols for complex terms
# P, Q, R, S, T, U, V, W = symbols('P Q R S T U V W')
#
# # calculate the difference
# X_diff = X - X_OT
# Y_diff = Y - Y_OT
#
# # Substitute complex terms with new symbols
# term1 = ((X - X_OT) * P + (Y - Y_OT) * Q - R) ** 2
# term2 = ((X - X_OT) * S + (Y - Y_OT) * T + U) ** 2
# term3 = ((X - X_OT) * V + (Y - Y_OT) * W - Z_OT) ** 2
#
# # calculate the result
# result = term1 / a ** 2 + term2 / b ** 2 + term3 / c ** 2 - 1
#
# # Expand the result
# result_expanded = expand(result)
#
# # Collect coefficients for X^2, Y^2, XY, and constant term
# A = result_expanded.coeff(X, 2)
# B = result_expanded.coeff(X * Y)
# C = result_expanded.coeff(Y, 2)
#
# result_expanded = result_expanded - A * X ** 2 - B * X * Y - C * Y ** 2
# result_expanded = result_expanded.expand()
# print(f"result_expanded: {result_expanded.simplify()}")
# D = result_expanded.coeff(X).simplify()
# E = result_expanded.coeff(Y).simplify()
#
# # Find the constant F (all terms that don't contain X or Y)
# F = result_expanded - A * X ** 2 - B * X * Y - C * Y ** 2
# F_const = F.subs(X, 0).subs(Y, 0)
#
# # Redefine F as an expression with simplified symbols
# F = F_const.expand().simplify()
#
# print(f"A: {A}")
# print(f"B: {B}")
# print(f"C: {C}")
# print(f"D: {D}")
# print(f"E: {E}")
# print(f"F: {F}")
# #
# # # print the results
# print(f"A: {simplify(A)}")
# print(f"B: {simplify(B)}")
# print(f"C: {simplify(C)}")
# print(f"D: {simplify(D)}")
# print(f"E: {simplify(E)}")
# print(f"F: {simplify(F)}")
import numba


@numba.njit
def calculate_terms(X_OT, Y_OT, Z_OT, a, b, c, psi_T, eps_T, omega_T):
    # Calculate the sin and cos once
    sin_psi, cos_psi = np.sin(psi_T), np.cos(psi_T)
    sin_omega, cos_omega = np.sin(omega_T), np.cos(omega_T)
    sin_eps, cos_eps = np.sin(eps_T), np.cos(eps_T)

    # Calculate the complex terms only once
    sin_psi_cos_omega = sin_psi * cos_omega
    sin_psi_sin_omega = sin_psi * sin_omega
    cos_psi_cos_omega = cos_psi * cos_omega
    cos_psi_sin_omega = cos_psi * sin_omega
    sin_eps_sin_psi = sin_eps * sin_psi
    cos_eps_sin_psi = cos_eps * sin_psi
    sin_eps_cos_psi = sin_eps * cos_psi
    cos_eps_cos_psi = cos_eps * cos_psi

    # Calculate the terms
    P = cos_psi_cos_omega
    Q = -cos_psi_sin_omega
    R = -sin_psi
    S = cos_eps * sin_omega + sin_eps_sin_psi * cos_omega
    T = cos_eps * cos_omega - sin_eps_sin_psi * sin_omega
    U = sin_eps_cos_psi
    V = sin_eps * sin_omega - cos_eps_sin_psi * cos_omega
    W = sin_eps * cos_omega + cos_eps_sin_psi * sin_omega

    # Precompute some coefficients
    a_sq = a ** 2
    b_sq = b ** 2
    c_sq = c ** 2
    abc_sq = a_sq * b_sq * c_sq

    # Compute grouped terms
    A = (P ** 2) / a_sq + (S ** 2) / b_sq + (V ** 2) / c_sq
    B = 2 * ((P * Q) / a_sq + (S * T) / b_sq + (V * W) / c_sq)
    C = (Q ** 2) / a_sq + (T ** 2) / b_sq + (W ** 2) / c_sq

    D = 2 * (-P * b_sq * c_sq * (P * X_OT + Q * Y_OT + R)
             + S * a_sq * c_sq * (-S * X_OT - T * Y_OT + U)
             - V * a_sq * b_sq * (V * X_OT + W * Y_OT + Z_OT)) / abc_sq

    E = 2 * (-Q * b_sq * c_sq * (P * X_OT + Q * Y_OT + R)
             + T * a_sq * c_sq * (-S * X_OT - T * Y_OT + U)
             - W * a_sq * b_sq * (V * X_OT + W * Y_OT + Z_OT)) / abc_sq

    F = (-abc_sq + a_sq * b_sq * (V ** 2 * X_OT ** 2 + 2 * V * W * X_OT * Y_OT +
                                  2 * V * X_OT * Z_OT + W ** 2 * Y_OT ** 2 +
                                  2 * W * Y_OT * Z_OT + Z_OT ** 2) +
         a_sq * c_sq * (S ** 2 * X_OT ** 2 + 2 * S * T * X_OT * Y_OT -
                        2 * S * U * X_OT + T ** 2 * Y_OT ** 2 -
                        2 * T * U * Y_OT + U ** 2) +
         b_sq * c_sq * (P ** 2 * X_OT ** 2 + 2 * P * Q * X_OT * Y_OT +
                        2 * P * R * X_OT + Q ** 2 * Y_OT ** 2 +
                        2 * Q * R * Y_OT + R ** 2)) / abc_sq

    return A, B, C, D, E, F


def calc_a(lambda1, det_m0, det_m):
    return np.sqrt(-det_m0 / (det_m * lambda1))


def calc_b(lambda2, det_m0, det_m):
    return np.sqrt(-det_m0 / (det_m * lambda2))


def ellipse_of_intersection(nx, ny, nz, f, sx, sy, sz):
    # ff = f
    # swap the largest normal component to nz
    # normal = np.array([nx, ny, nz])
    # indices = np.abs(normal).argsort()
    # nx, ny, nz = normal[indices]
    # ensure the plane normal vector is normalized
    k = np.linalg.norm(np.array([nx, ny, nz]))
    # print(f"k: {k}")
    nx, ny, nz = np.array([nx, ny, nz]) / np.linalg.norm(np.array([nx, ny, nz]))
    # f = f / k
    A = 1 / sx ** 2 + nx ** 2 / (nz ** 2 * sz ** 2)
    B = 2 * nx * ny / (nz ** 2 * sz ** 2)
    C = 1 / sy ** 2 + ny ** 2 / (nz ** 2 * sz ** 2)
    D = 2 * nx * f / (nz ** 2 * sz ** 2)
    E = 2 * ny * f / (nz ** 2 * sz ** 2)
    F = f ** 2 / (nz ** 2 * sz ** 2) - 1.
    m0 = np.array([
        [F, D / 2, E / 2],
        [D / 2, A, B / 2],
        [E / 2, B / 2, C]
    ])
    m = np.array([
        [A, B / 2],
        [B / 2, C]
    ])

    # Print all values for verification
    print(f"{A=},\n{B=},\n{C=},\n{D=},\n{E=},\n{F=}")

    # Get eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(m)
    lambda1, lambda2 = sorted(eigvals)
    eigvec1, eigvec2 = eigvecs[:, eigvals.argmin()], eigvecs[:, eigvals.argmax()]

    print(eigvec1, eigvec2)
    print(lambda1, lambda2)
    # Get theta from the eigenvector corresponding to the largest eigenvalue
    theta = np.arccos(np.dot(eigvec1, np.array([1, 0]) / np.linalg.norm(eigvec1)))
    print(theta)

    # Calculate the parameters a0 and b0
    det_m0 = np.linalg.det(m0)
    det_m = np.linalg.det(m)

    # Calculate the parameters a0 and b0
    a0 = calc_a(lambda1, det_m0, det_m)
    b0 = calc_b(lambda2, det_m0, det_m)
    print(f"{a0=},\n{b0=},\n{theta=}\n")

    # Calculate the center of the ellipse
    denominator = 4 * A * C - B ** 2
    x0 = (B * E - 2 * C * D) / denominator
    y0 = (B * D - 2 * A * E) / denominator
    z0 = - (nx * x0 + ny * y0 + f) / nz

    print(f"{x0=},\n{y0=},\n{z0=}\n")
    print(f"{a0=},\n{b0=},\n{theta=}\n")

    # Calculate the angles theta, phi and omega
    qx = nx * (f / k if f != 0 else 1)
    qy = ny * (f / k if f != 0 else 1)
    qz = nz * (f / k if f != 0 else 1)

    print(f"{qx=},\n{qy=},\n{qz=}\n")

    # theta = np.arctan2(normalized_b, normalized_a - normalized_c) / 2
    phi = np.pi / 2 - np.arctan2(np.abs(qz), np.sqrt(qx ** 2 + qy ** 2))
    # phi = np.arctan2(np.abs(qz), np.sqrt(qx ** 2 + qy ** 2))
    omega = np.pi - np.arctan2(qy, qx)

    print(f"{np.rad2deg(phi)=},\n{np.rad2deg(omega)=}\n")

    # Create transformation matrix
    t = np.eye(4)
    R = rot3(-omega) @ rot2(-phi)
    t[:3, :3] = R
    t[:3, 3] = np.array([x0, y0, z0])

    eps_t = - np.arctan(R[1, 2] / R[2, 2])
    psi_t = np.arcsin(R[0, 2])
    omega_t = - np.arctan2(R[0, 1], R[0, 0])

    t_inv = np.linalg.inv(t)
    print(t)
    print(t_inv)

    x0_t, y0_t, z0_t = t_inv[:3, 3]

    print(f"{np.rad2deg(psi_t)=},\n{np.rad2deg(eps_t)=},\n{np.rad2deg(omega_t)=}\n")

    A_inv, B_inv, C_inv, D_inv, E_inv, F_inv = calculate_terms(x0_t, y0_t, z0_t, a, b, c, psi_t, eps_t, omega_t)

    # F_inv, E_inv, D_inv = F_inv + E_inv + D_inv, 0, 0

    print(f"{A_inv=},\n{B_inv=},\n{C_inv=},\n{D_inv=},\n{E_inv=},\n{F_inv=}")

    # # Form new m0 and m matrices
    m0_inv = np.array([
        [F_inv, D_inv / 2, E_inv / 2],
        [D_inv / 2, A_inv, B_inv / 2],
        [E_inv / 2, B_inv / 2, C_inv]
    ])
    m_inv = np.array([
        [A_inv, B_inv / 2],
        [B_inv / 2, C_inv]
    ])

    # Get eigenvalues and eigenvectors
    eigvals_inv, eigvecs_inv = np.linalg.eig(m_inv)
    lambda1_inv, lambda2_inv = sorted(eigvals_inv)

    # Calculate the parameters ai and bi
    det_m0_inv = np.linalg.det(m0_inv)
    det_m_inv = np.linalg.det(m_inv)
    ai = calc_a(lambda1_inv, det_m0_inv, det_m_inv)
    bi = calc_b(lambda2_inv, det_m0_inv, det_m_inv)
    theta_inv = np.arccos(np.dot(eigvecs_inv[:, eigvals_inv.argmin()],
                                 np.array([1, 0]) / np.linalg.norm(eigvecs_inv[:, eigvals_inv.argmin()])))
    print(f"{ai=},\n{bi=},\n{theta_inv=}\n")

    # Pole of the plane
    px = sx ** 2 * nx / f
    py = sy ** 2 * ny / f
    pz = sz ** 2 * nz / f
    pole = -np.array([px, py, pz])

    return (a0, b0, ai, bi), (x0, y0, z0), t, t_inv, theta, theta_inv, pole


# def ellipse_of_intersection(a, b, c, λ, µ, ν, h):
#     # Compute the coefficients for the ellipse equation
#     A = (ν ** 2 * c ** 2 / a ** 2) + λ ** 2
#     B = 2 * λ * µ
#     C = (ν ** 2 * c ** 2 / b ** 2) + µ ** 2
#     D = -2 * h * λ
#     E = -2 * h * µ
#     F = h ** 2 - ν ** 2 * c ** 2
#
#     print(f"{A=}, {B=}, {C=}, {D=}, {E=}, {G=} ")
#
#     M0 = np.array([[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, G]])
#
#     # Calculate the center of the ellipse
#     v = (B * E - 2 * C * D) / (4 * A * C - B ** 2)
#     w = (B * D - 2 * A * E) / (4 * A * C - B ** 2)
#
#     return np.array([v, w, (h - λ * ν - µ * w) / ν])
#
#
# def ellipse_of_intersection(a, b, c, λ, µ, ν, h):
#     # Compute the coefficients for the ellipse equation
#     A = 1 / (a ** 2) + (λ ** 2) / (ν ** 2 * c ** 2)
#     B = (2 * λ * µ) / (ν ** 2 * c ** 2)
#     C = 1 / (b ** 2) + (µ ** 2) / (ν ** 2 * c ** 2)
#     D = -2 * h * λ
#     E = -2 * h * µ
#     F = h ** 2 - ν ** 2 * c ** 2
#
#     print(f"{A=}, {B=}, {C=}, {D=}, {E=}, {F=}")
#
#     # The matrix M0
#     M0 = np.array([[F, D / 2, E / 2], [D / 2, A, B / 2], [E / 2, B / 2, C]])
#
#     # The matrix M
#     M = np.array([[A, B / 2], [B / 2, C]])
#
#     # Compute eigenvalues and eigenvectors
#     eigvals, eigvecs = np.linalg.eig(M)
#
#     # Compute semi-axes of the ellipse
#     a0 = np.sqrt(np.linalg.det(M0) / np.linalg.det(M) * eigvals[0])
#     b0 = np.sqrt(np.linalg.det(M0) / np.linalg.det(M) * eigvals[1])
#
#     print(f"Semi-axes: {a0}, {b0}")
#
#     # Calculate the center of the ellipse
#     v = (B * E - 2 * C * D) / (4 * A * C - B ** 2)
#     w = (B * D - 2 * A * E) / (4 * A * C - B ** 2)
#
#     return np.array([v, w, (h - λ * v - µ * w) / ν])

# # Compute the lengths of the semi-major and semi-minor axes
# numerator = 2 * (A * E ** 2 + C * D ** 2 - B * D * E + (B ** 2 - 4 * A * C) * G)
# denominator_a = B ** 2 - 4 * A * C
# denominator_b = (A + C) + np.sqrt((A - C) ** 2 + B ** 2)
# a = np.sqrt(numerator / (denominator_a * denominator_b))
# b = np.sqrt(numerator / (denominator_a * (2 * (A + C) - denominator_b)))
#
# # Calculate the rotation angle
# if B == 0 and A < C:
#     θ = 0
# elif B == 0 and A > C:
#     θ = np.pi / 2
# else:
#     θ = np.arctan(2 * B / (A - C)) / 2
#     if A > C:
#         θ += np.pi / 2
#
# return A, B, C, D, E, G, v, w, a, b, θ


if __name__ == '__main__':
    from matplotlib import rc

    rc('text', usetex=True)  # To enable the use of LaTeX text formatting
    dark_mode = True
    limit = 2.0
    from pydin.core.gravitation import PolyhedronShape

    from pydin.core.shape import sphere, ellipsoid

    Sphere = sphere
    Ellipsoid = ellipsoid

    # enhance_classes_with_show()
    #
    # s = Sphere(radius=1.0)
    # e = Ellipsoid(*(3 * np.array([1.0, 0.2, 2.0])))
    # e.show()

    # Now you can call show() on these instances
    # sphere_instance.show()

    # polyhedron_shape = PolyhedronShape(
    #     *define_geometry(
    #         file_path='tests/data/Eros.node',
    #         file_faces='tests/data/Eros.face'
    #     )
    # )

    # # easy to create.
    nodes, faces = define_geometry(
        file_path='tests/data/Eros.node',
        file_faces='tests/data/Eros.face'
    )

    model_eros = PolyhedronShape(nodes, faces)
    # model_pyramid = PolyhedralGravity(*get_pyramid(), density=2e3)
    #
    # model = model_eros
    #
    # # various optimal methods
    # v = model.potential(position)          # scalar
    # vv = model.potential_series(position)  # vector [1, m]
    # vvv = model.potential_grid(position)   # matrix [m, n]

    # polyhedron_shape = get_pyramid(base_length=limit / 2, height=limit / 2)

    # polyhedron_shape = get_pyramid(base_length=limit / 2, height=limit / 2)

    # polyhedron_shape = PolyhedronShape(polyhedron_shape.nodes, polyhedron_shape.faces)
    # polyhedron_shape = get_right_angled_triangular_prism(limit / 2)
    # polyhedron_shape = get_cube(limit)

    # pyramid = get_pyramid(base_length=limit / 2, height=limit / 2)
    # pyramid.reorient_faces("cw")
    # nodes, faces = pyramid.nodes, pyramid.faces
    # polyhedron_shape = PolyhedronShape(nodes, faces)
    # polyhedron_shape = PolyhedronShape(['tests/data/Eros.node', 'tests/data/Eros.face'])

    import numpy as np

    # # Define the radii of the ellipsoid
    # radii = np.array([3.0, 2.0, 1.0])
    #
    # # Create a unit sphere
    # sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    #
    # # Scale it to form the ellipsoid
    # ellipsoid_mesh = sphere.apply_transform(np.diag(np.append(radii, 1)))
    #
    # nodes = ellipsoid_mesh.vertices
    # faces = ellipsoid_mesh.faces

    # polyhedron_shape = model_eros.reorient_faces("cw")
    # nodes = polyhedron_shape.get_nodes()
    # faces = polyhedron_shape.get_faces()

    # assert polyhedron_shape.are_normals_outward_pointing() and polyhedron_shape.are_triangles_not_degenerated()

    # nodes, faces = get_right_angled_triangular_prism(limit / 2)

    n = 50j
    n_quiver = 12j
    # nodes, faces = get_right_angled_triangular_prism(1.)

    # # Assume nodes and faces are defined
    # nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    # faces = np.array([[0, 1, 2], [0, 2, 3]])

    # plot_3d_mesh_cross_section('3D Mesh', nodes, faces, limit=limit, dark_mode=dark_mode)

    # plot_3d_mesh_mayavi('3D Mesh', nodes, faces, dark_mode=dark_mode, filename='3D_mesh_mayavi.png', show=True,
    #                     plane_origin=(0, 0, 0), limit=limit)

    # xv, yv, zv = define_spatial_grid_z0(limit=limit, n_points=100)

    # def calculate_potential(xv, yv, zv, nodes, faces, density=2675.0):
    #     polyhedron = Polyhedral(nodes, faces, density)
    # return polyhedron.potential_grid((xv, yv, zv))
    limit = 10.0
    figure = mlab.figure('3D Contour', bgcolor=(0.2, 0.2, 0.2), fgcolor=(1., 1., 1.), size=(1000, 800))
    a = 4.
    b = 3.
    c =2.

    l = 0
    m = 0
    n = 1
    # l, m, n = np.array([l, m, n]) / np.linalg.norm([l, m, n])

    f = 0.2
    (a0, b0, ai, bi), (x0, y0, z0), T, T_inv, theta, theta_inv, pole = ellipse_of_intersection(l, m, n, f, a, b, c)

    # plot_ellipsoid_mayavi(a, b, c, figure=figure, color=(1, 0, 0), opacity=0.8)

    t = np.linspace(0, 2 * np.pi, 100)
    x = ai * np.cos(t)
    y = bi * np.sin(t)
    # rotate x and y by theta_inv
    z = np.zeros_like(t)
    x, y, z = np.dot(rot3(theta_inv), np.array([x, y, z]))

    # # # Apply rotation using broadcasting
    # x, y, z = np.dot(rot3(-theta), np.array([x, y, z]))

    # Combine x, y, z into a 3xN array
    points = np.array([x, y, z, np.ones_like(x)])

    # Apply rotation using broadcasting
    rotated_points = np.dot(np.linalg.inv(T_inv), points)

    # Extract the rotated x, y, z coordinates
    x_rot, y_rot, z_rot, aa = rotated_points

    mlab.points3d(x0, y0, z0, color=(1, 1, 1), scale_factor=0.5)
    mlab.points3d(*pole, color=(1, 1, 1), scale_factor=0.5)
    mlab.plot3d(x_rot, y_rot, z_rot, color=(1, 1, 1))
    # mlab.plot3d(x, y, z, color=(1, 0, 0))
    # plot plane Ax + By + Cz + D = 0
    # Define ranges for x and y
    x = np.linspace(-limit, limit, 10)
    y = np.linspace(-limit, limit, 10)

    # Make them into a grid
    X, Y = np.meshgrid(x, y)

    # Ax * X + Ay * Y + Az * Z + Ad = 0

    # Calculate corresponding Z
    Z = -(l * X + m * Y + f) / n

    # Plot the surface
    mlab.mesh(X, Y, Z, color=(0, 0, 1), opacity=0.2)

    # u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    # x = a * np.cos(u) * np.sin(v)
    # y = b * np.sin(u) * np.sin(v)
    # z = c * np.cos(v)
    # mlab.mesh(x, y, z, color=(0, 1, 0), opacity=0.2)
    plot_ellipsoid_mayavi(a, b, c, figure=figure, color=(1, 0, 0), opacity=0.8)

    mlab.show()
    raise SystemExit

    # parametrise and plot an ellipsoid

    # Enable anti-aliasing
    # ellipsoid_instance.show()

    model_type = TriAxialEllipsoid
    # model_type = Polyhedral
    # nodes, faces = ellipsoid_mesh(a, b, c, 2)

    model_params = {
        TriAxialEllipsoid: {
            'a': 200.0,
            'b': 200.0,
            'c': 200.0,
            'mu': mu
        },
        Polyhedral: {
            'nodes': nodes,
            'faces': faces,
            'density': 10000.0
        }
    }
    a, b, c = model_params[TriAxialEllipsoid]['a'], model_params[TriAxialEllipsoid]['b'], \
        model_params[TriAxialEllipsoid]['c']
    if model_type == Polyhedral:
        nodes = model_params[Polyhedral]['nodes']
        faces = model_params[Polyhedral]['faces']
        shape = PolyhedronShape(nodes, faces)
        shape = shape.reorient_faces("cw")
        model_params[Polyhedral]['nodes'] = shape.get_nodes()
        model_params[Polyhedral]['faces'] = shape.get_faces()
        model = Polyhedral(**model_params[Polyhedral])
        limit = 1.2 * (np.max(nodes) - np.min(nodes))
        # OBJECT MESH ############
        plot_3d_mesh_mayavi(
            nodes,
            faces,
            figure=figure,
            filename='3D_mesh_mayavi_test.png',
            opacity=0.8,
            color=(0.5, 0.5, 1),
            n_limit=5,
            azimuth=45,
            elevation=45,  # what comes after azimuth and elevation, why of course, the roll
            dark_mode=True)

    elif model_type == TriAxialEllipsoid:
        a = model_params[TriAxialEllipsoid]['a']
        b = model_params[TriAxialEllipsoid]['b']
        c = model_params[TriAxialEllipsoid]['c']
        model = TriAxialEllipsoid(**model_params[TriAxialEllipsoid])
        limit = 1.2 * np.max([a, b, c])
        # create ellipsoid mesh
        plot_ellipsoid_mayavi(a, b, c,
                              figure=figure,
                              # wireframe=True,
                              color=(0.5, 0.5, 1),
                              opacity=0.8,
                              # dark_mode=True,
                              # add texture mapping to ~/Desktop/download.jpeg
                              texture='~/Desktop/download.jpeg',
                              )

    # plot_boundary_limit(figure, limit, shape='box', color=(1, 1, 1), opacity=0.2)
    plot_boundary_limit(figure, limit, shape='square', color=(1, 1, 1), opacity=0.8)

    from pydin.core.shape import ellipsoid, sphere

    n_points = 2000
    example_ellipsoid = ellipsoid(model_params[TriAxialEllipsoid]['a'],
                                  model_params[TriAxialEllipsoid]['b'],
                                  model_params[TriAxialEllipsoid]['c'])
    example_sphere = sphere((1000 + 200 + 200) / 3)


    # example_ellipsoid = example_ellipsoid.reorient_faces("cw")

    def generate_rays(n_points, origin):
        rays = np.random.uniform(-1, 1, (3, n_points))
        origins = np.tile(origin, (n_points, 1)).T
        return origins, rays


    def filter_intersections(origins, intersections):
        valid_indices = ~np.isnan(intersections).any(axis=0)
        valid_origins = origins[:, :len(valid_indices)]  # trim origins to match the size of valid_indices
        return valid_origins[:, valid_indices], intersections[:, valid_indices]


    def plot_intersections(origins, intersections, color):
        mlab.points3d(intersections[0], intersections[1], intersections[2], color=color, scale_factor=8)


    def plot_rays_from_origin(origins, intersections, color, scale_factor=1, opacity=0.5):
        origins, intersections = filter_intersections(origins, intersections)
        mlab.quiver3d(origins[0, :], origins[1, :], origins[2, :],
                      intersections[0, :] - origins[0, :],
                      intersections[1, :] - origins[1, :],
                      intersections[2, :] - origins[2, :],
                      color=color, scale_factor=scale_factor, mode='2ddash', opacity=opacity)


    # generate rays from the inside of the ellipsoid going out, sample, and calculate the intersections
    # with the ellipsoid
    dist = 2500
    origin = np.array([0, dist, 0])
    up = np.array([0.5, 0, 0.5], dtype=np.float64)
    target = np.array([0, 0, 0], dtype=np.float64)

    rays = np.random.uniform(-1, 1, (3, n_points))
    origins = np.tile(origin, (n_points, 1)).T

    # intersects = np.array([example_ellipsoid.ray_intersection(origin, ray) for origin, ray in zip(origins, rays)])

    # pos = np.array([0, 2000, 0])
    # target = np.array([0, 0, 0])
    # up = np.array([0, 0, 1])
    # hfov = 30
    aspect = 1
    near = 0
    far = dist
    #
    pos = origin.astype(np.float64)
    hfov = 10

    # frustum_coords = get_frustum_coords(pos, target, up, hfov, aspect, near, far)
    # pixel_directions = get_pixel_directions(pos, frustum_coords, h_resolution=800, v_resolution=600).T
    # print(pixel_directions)
    # plot_frustum(pos, target, up, hfov, aspect, near, far)
    #
    # pixel_origins = np.tile(origin, (n_points, 1)).T
    # # pixel_origins = np.tile(pos.reshape(3, 1), (1, pixel_directions.shape[1]))  # Now shape is (3, n)
    # intersects_ellipsoid = example_ellipsoid.ray_intersection_series2(pixel_origins, pixel_directions)
    # plot_rays_from_origin(pixel_origins, intersects_ellipsoid, color=(1, 0, 0))

    frustum_coords = get_frustum_coords(pos, target, up, hfov, aspect, near, far)
    pixel_directions = get_pixel_directions(pos, target, up, hfov, aspect, near, far, h_resolution=50, v_resolution=50)
    # pixel_directions = pixel_directions.reshape(3, -1)
    # pixel_directions2 = pixel_directions * 1000
    # pixel_directions = pixel_directions.reshape(3, -1)

    # mlab.plot3d(pixel_directions[0, :], pixel_directions[1, :], pixel_directions[2, :], color=(1, 0, 0),
    #             tube_radius=0.1)
    plot_frustum(pos, target, up, hfov, aspect, near, far)

    pixel_origins = np.tile(origin, (pixel_directions.shape[1], 1)).T
    print(pixel_origins.shape)
    print(pixel_directions.shape)
    intersects_ellipsoid = example_ellipsoid.ray_intersection_series2(pixel_origins, pixel_directions)
    print(intersects_ellipsoid.shape)
    cyan = (0, 1, 1)
    plot_rays_from_origin(pixel_origins, intersects_ellipsoid, color=cyan, scale_factor=1, opacity=0.3)
    mlab.points3d(intersects_ellipsoid[0, :], intersects_ellipsoid[1, :], intersects_ellipsoid[2, :], color=(1, 1, 1),
                  scale_factor=3)

    a = model_params[TriAxialEllipsoid]['a']
    b = model_params[TriAxialEllipsoid]['b']
    c = model_params[TriAxialEllipsoid]['c']

    # plot voxel grid for marching cubes
    voxel_grid = np.mgrid[-a:a:50j, -b:b:50j, -c:c:50j]

    # mask those that are not example_ellipsoid.is_inside
    # voxel_grid = voxel_grid[:, example_ellipsoid.is_inside(voxel_grid)]
    xyz = voxel_grid.reshape(3, -1)
    mask = example_ellipsoid.is_inside(xyz)
    mask = np.array(mask)
    xyz_outside = xyz[:, ~mask]
    xyz_inside = xyz[:, mask]
    x = xyz_inside[0, :]
    y = xyz_inside[1, :]
    z = xyz_inside[2, :]
    # mlab.points3d(x, y, z, color=(0, 1, 0), scale_factor=2, opacity=0.8)

    # x_out = xyz_outside[0, :]
    # y_out = xyz_outside[1, :]
    # z_out = xyz_outside[2, :]
    # mlab.points3d(x_out, y_out, z_out, color=(1, 0, 0), scale_factor=2, opacity=0.8)

    # plot_rays_from_origin(origins, intersects_ellipsoid, color=(1, 0, 0), scale_factor=1)

    # Compute a_prime, b_prime (this should be done in your C++ code)
    # a_prime, b_prime = example_ellipsoid.get_silhouette_dimensions(pos, target)

    # mlab.quiver3d(origins_sphere[0, :], origins_sphere[1, :], origins_sphere[2, :],
    #               intersects_sphere[0, :] - origins_sphere[0, :],
    #               intersects_sphere[1, :] - origins_sphere[1, :],
    #               intersects_sphere[2, :] - origins_sphere[2, :],
    #               color=(0, 1, 0), scale_factor=1)
    # # draw lines between the intersections on sphere and ellipsoid
    # difference = intersects_ellipsoid - intersects_sphere
    # mlab.quiver3d(intersects_sphere[0, :], intersects_sphere[1, :], intersects_sphere[2, :],
    #               difference[0, :], difference[1, :], difference[2, :], color=(0, 0, 1), scale_factor=1)

    # mlab.show()

    # # sample quiver for acceleration
    xv, yv, zv = np.mgrid[
                 -limit:limit:n_quiver,
                 -limit:limit:n_quiver,
                 -limit:limit:n_quiver]  # spatial grid
    # avx, avy, avz = calculate_acceleration(xv, yv, zv, nodes, faces, density=1)
    # avx, avy, avz = model.acceleration_grid((xv, yv, zv))
    # plot_3d_quiver_mayavi(xv,
    #                       yv,
    #                       zv,
    #                       avx,
    #                       avy,
    #                       avz,
    #                       figure=figure,
    #                       scale_factor=0.05,
    #                       color=(0, 0, 1))

    # The grid is 3D but we're interested only in the x-y plane
    xv, yv, zv = np.mgrid[-limit:limit:n, -limit:limit:n, -0:0:1j]  # spatial grid

    vv = model.potential_grid((xv, yv, zv))
    plot_3d_contour_mayavi(xv,
                           yv,
                           zv,
                           vv,
                           contours=80,
                           figure=figure,
                           cmap='inferno',
                           dark_mode=True,
                           show=False,
                           save=False,
                           axes=False,
                           azimuth=45,
                           elevation=45,  # what comes after azimuth and elevation, why of course, the roll

                           )

    mlab.view(azimuth=45, elevation=45)

    # Get the VTK render window
    figure.scene.render_window.point_smoothing = True
    figure.scene.render_window.line_smoothing = True
    figure.scene.render_window.polygon_smoothing = True
    figure.scene.render_window.multi_samples = 16  # Try with 4 if you think this is slow

    # Enable anti-aliasing

    # Refresh the scene to apply the change
    # mlab.gcf().scene.render()
    # mlab.gcf().scene.render()

    # show the figure
    mlab.show()

    # XZ GRID SAMPLE ############
    # xv, yv, zv = np.mgrid[-limit:limit:n, -0:0:1j, -limit:limit:n]  # spatial grid
    # vv = calculate_potential(xv, yv, zv, nodes, faces, density=1)
    # plot_3d_contour_mayavi(
    #     xv,
    #     yv + limit,
    #     zv,
    #     vv,
    #     figure=figure,
    #     cmap='inferno',
    #     dark_mode=True,
    #     show=False,
    #     save=False)
    #
    # # YZ GRID SAMPLE ############
    # xv, yv, zv = np.mgrid[-0:0:1j, -limit:limit:n, -limit:limit:n]  # spatial grid
    # vv = calculate_potential(xv, yv, zv, nodes, faces, density=1)
    # plot_3d_contour_mayavi(
    #     xv + limit,
    #     yv,
    #     zv,
    #     vv,
    #     figure=figure,
    #     cmap='inferno',
    #     dark_mode=True,
    #     show=False,
    #     save=False)

    # xv, yv, zv = define_spatial_grid_z0(limit=limit, n_points=20)

    # plot_3d_mesh('3D Mesh', nodes, faces)

    # if not check_face_orientation(nodes, faces):
    #     print("Faces are not correctly oriented. Reorienting...")
    #     nodes, faces = reorient_faces(nodes, faces)

    # xv, yv, zv = define_spatial_grid_z0(limit=limit, n_points=100)
    # poly2d_quiver(xv, yv, zv, nodes, faces, density=1, dark_mode=dark_mode)
    #
    # xv, yv, zv = define_spatial_grid_z0(limit=limit, n_points=20)
    # poly2d_contour(xv, yv, zv, nodes, faces, density=1, dark_mode=dark_mode)
    #

    # test_polyhedral_plot()
    # poly_quiver()
    # test_potential_series()
    # test_acceleration_series()
    # test_potential_grid()
    # test_acceleration_grid()
    # test_acceleration_grid_3d()
    # animation_test()

# # Bazel sets this environment variable
# # runfiles_dir = os.environ.get('RUNFILES_DIR')
# # eros_50k_path = os.path.join(runfiles_dir, 'eros_50k_ply/file/eros_50k.ply')
#     print(runfiles_dir)
#     print(eros_50k_path)
#     gravity = Polyhedral(eros_50k_path, 2.5)
