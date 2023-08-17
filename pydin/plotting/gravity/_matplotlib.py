__requires__ = ["matplotlib", "numpy"]

__all__ = [
    'plot_quiver_2d',
    'plot_contour_2d',
    'plot_3d_mesh',
]

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_figure_setup(title, xv, yv, data, cmap='jet', dark_mode=True, figsize=(10, 10), fig=None, ax=None,
                      gridlines=True):
    """ Set up the plot figure and axes with proper settings """

    # Set style based on dark_mode flag
    plt.style.use('dark_background' if dark_mode else 'default')
    cmap = 'inferno' if dark_mode else cmap

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Set plot title and axes labels
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('$X$', fontsize=15)
    ax.set_ylabel('$Y$', fontsize=15)

    # Set tick properties
    ax.tick_params(axis='both', which='major', labelsize=10, color='white' if dark_mode else 'black')
    ax.tick_params(axis='both', which='minor', labelsize=8, color='white' if dark_mode else 'black')

    # Set minor locator
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Set major ticks visible
    ax.xaxis.get_major_ticks()[0].label1.set_visible(True)
    ax.yaxis.get_major_ticks()[0].label1.set_visible(True)

    # Set gridlines
    if gridlines:
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_aspect('equal')

    return fig, ax, cmap


def create_colorbar(fig, ax, mappable, label, log=False):
    """ Create a color bar for the plot """

    # Create inset axes for the colorbar
    axins = inset_axes(ax, width="5%", height="100%", loc='lower left', bbox_to_anchor=(1.07, 0., 1, 1),
                       bbox_transform=ax.transAxes, borderpad=0)

    # Set up the formatter
    format = ticker.ScalarFormatter()
    format.set_scientific(True)
    format.set_powerlimits((-1, 1))
    format.set_useMathText(True)

    # Add color bar
    cbar = fig.colorbar(mappable, cax=axins, format=format)
    cbar.set_label(label, fontsize=15, labelpad=20)
    return cbar


def plot_quiver_2d(xv, yv, av_x, av_y, title=r'Quiver plot of acceleration at $Z=0$', cmap='jet',
                   dark_mode=True, show=False, log=False, save=True, filename='2D_acceleration_vectors.png',
                   fig=None, ax=None):
    """ Create a quiver plot """

    # Calculate magnitudes and normalize vectors
    mag = np.sqrt(av_x ** 2 + av_y ** 2)
    av_x_normalized = av_x / mag
    av_y_normalized = av_y / mag

    cbar_label = r'$\log_{10}(\|\mathbf{a}\|)$' if log else r'$\|\mathbf{a}\|$'
    if log:
        mag = np.log10(mag)

    # Set up plot
    fig, ax, cmap = plot_figure_setup(title, xv, yv, mag, cmap, dark_mode, fig=fig, ax=ax)

    # Create quiver
    quiver = ax.quiver(xv[:, :, 0], yv[:, :, 0], -av_x_normalized[:, :, 0], -av_y_normalized[:, :, 0],
                       mag[:, :, 0], cmap=cmap, lw=1, pivot='middle', scale=30.0)

    # Add color bar
    cbar = create_colorbar(fig, ax, quiver, cbar_label)

    # Save and show plot as per the flags
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    return fig, ax


def plot_contour_2d(xv, yv, potentials, title=r'Contour plot of potential at $Z=0$', cmap='viridis',
                    dark_mode=True, show=False, log=False, save=True, filename='2D_potential_contour.png',
                    fig=None, ax=None):
    """ Create a contour plot """

    cbar_label = r'$\log_{10}(V)$' if log else r'$V$'
    if log:
        potentials = np.log10(potentials)

    # Set up plot
    fig, ax, cmap = plot_figure_setup(title, xv, yv, potentials, cmap, dark_mode, fig=fig, ax=ax)

    # Create contour plot
    contour = ax.contourf(xv[:, :, 0], yv[:, :, 0], potentials[:, :, 0], cmap=cmap)

    # Add color bar
    cbar = create_colorbar(fig, ax, contour, cbar_label)

    # Save and show plot as per the flags
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    return fig, ax


def plot_3d_mesh(title, nodes, faces, cmap='jet', dark_mode=True, figsize=(10, 10),
                 show=False, save=True, filename='3D_mesh.png'):
    """ Create a 3D mesh plot """

    # Set style based on dark_mode flag
    plt.style.use('dark_background' if dark_mode else 'default')
    cmap = 'inferno' if dark_mode else cmap

    # Set up the figure and axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Set plot title and axes labels
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    ax.set_zlabel('Z', fontsize=15)

    # Set tick properties
    ax.tick_params(axis='both', which='major', labelsize=10, color='white' if dark_mode else 'black')
    ax.tick_params(axis='both', which='minor', labelsize=8, color='white' if dark_mode else 'black')

    # Set minor locator
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.zaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Set major ticks visible
    ax.xaxis.get_major_ticks()[0].label1.set_visible(True)
    ax.yaxis.get_major_ticks()[0].label1.set_visible(True)
    ax.zaxis.get_major_ticks()[0].label1.set_visible(True)

    # Create a 3D mesh using Poly3DCollection
    poly3d = [nodes[faces[i]] for i in range(len(faces))]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    # Calculate the limits and set them
    all_nodes = nodes[faces.ravel()]
    ax.set_xlim(all_nodes[:, 0].min(), all_nodes[:, 0].max())
    ax.set_ylim(all_nodes[:, 1].min(), all_nodes[:, 1].max())
    ax.set_zlim(all_nodes[:, 2].min(), all_nodes[:, 2].max())

    # Adjust the viewing angle
    ax.view_init(elev=20, azim=-35)

    ax.set_aspect('auto')

    # Save and show plot as per the flags
    if save:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    return fig, ax
