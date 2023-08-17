from .. import attempt_import
from ..__mayavi import mayavi_style

mlab = attempt_import('mayavi.mlab')
np = attempt_import('numpy')


@mayavi_style(style='dark', plot_params={'filename': '3D_mesh.png'})
def plot_3d_mesh_mayavi(nodes, faces, **kwargs):
    figure = kwargs.get('figure', mlab.gcf())

    mesh = mlab.triangular_mesh(nodes[:, 0], nodes[:, 1], nodes[:, 2], faces,
                                color=kwargs.get('color', (0.5, 0.5, 0.5)),
                                opacity=kwargs.get('opacity', 0.6),
                                figure=figure, representation='surface', line_width=0.4, tube_sides=0.1)

    plane_origin = kwargs.get('plane_origin', None)
    if plane_origin is not None:
        cut_plane = mlab.pipeline.scalar_cut_plane(mesh, plane_orientation='z_axes', figure=figure)
        cut_plane.implicit_plane.origin = plane_origin

    mlab.view(azimuth=kwargs.get('azimuth', 180), elevation=kwargs.get('elevation', 90))

    limit = kwargs.get('limit', None)
    if limit is not None:
        x_lim_min, x_lim_max = -limit, limit
        y_lim_min, y_lim_max = -limit, limit
        z_lim_min, z_lim_max = -limit, limit
        mlab.points3d(np.array([x_lim_min, x_lim_max]),
                      np.array([y_lim_min, y_lim_max]),
                      np.array([z_lim_min, z_lim_max]),
                      mode='2dcross',
                      color=(0, 0, 0),
                      opacity=0, figure=figure)

    if kwargs.get('save', False):
        mlab.savefig(kwargs['filename'])
    if kwargs.get('show', False):
        mlab.show()

    return figure


# Your function implementation

@mayavi_style(style='dark', plot_params={'filename': '3D_acceleration_vectors.png'})
def plot_3d_quiver_mayavi(xv, yv, zv, av_x, av_y, av_z, **kwargs):
    # Calculate magnitude and normalize
    mag = np.sqrt(av_x ** 2 + av_y ** 2 + av_z ** 2)
    av_x_normalized = av_x / mag
    av_y_normalized = av_y / mag
    av_z_normalized = av_z / mag

    # Extract or define default parameters from kwargs
    colormap = kwargs.get('cmap', 'jet')
    view_params = kwargs.get('view', {'azimuth': 180, 'elevation': 90})
    filename = kwargs.get('filename', '3D_acceleration_vectors.png')
    show = kwargs.get('show', False)
    save = kwargs.get('save', True)

    # Create quiver plot
    mlab.quiver3d(
        xv, yv, zv,
        av_x_normalized,
        av_y_normalized,
        av_z_normalized,
        scalars=mag,
        colormap=colormap,
        mode='arrow',
        scale_factor=0.5
    )

    view = kwargs.get('view', {'azimuth': 180, 'elevation': 90})
    if view is not None:
        mlab.view(**view)

    # Save or display the figure
    if save:
        mlab.savefig(filename)
    if show:
        mlab.show()

    return mlab.gcf()


@mayavi_style(style='dark',
              plot_params={'filename': '3D_contour.png'},
              # uses_params=['figure']
              )
def plot_3d_contour_mayavi(xv, yv, zv, vv, **kwargs):
    figure = kwargs.get('figure', mlab.gcf())

    surf = mlab.contour3d(
        xv, yv, zv, vv,
        contours=kwargs.get('contours', 10),
        colormap=kwargs.get('cmap', 'jet'),
        figure=figure)

    if kwargs.get('axes', False):
        mlab.axes(figure=figure)

    if kwargs.get('colorbar', False):
        mlab.colorbar(surf, orientation='vertical')

    if kwargs.get('title', None) is not None:
        mlab.title(kwargs['title'], figure=figure)

    view = kwargs.get('view', {'azimuth': 180, 'elevation': 90})
    if view is not None:
        mlab.view(**view)

    if kwargs.get('save', True):
        filename = kwargs.get('filename', '3D_contour.png')
        mlab.savefig(filename)

    if kwargs.get('show', False):
        mlab.show()

    return figure


from itertools import combinations, product


@mayavi_style(style='dark', plot_params={'filename': 'boundary_limit.png'})
def plot_boundary_limit(figure, limit, shape='box', **kwargs):
    color = kwargs.get('color', (1, 1, 1))
    opacity = kwargs.get('opacity', 0.15)
    offset = kwargs.get('offset', 0.0)
    tube_radius = kwargs.get('tube_radius', None)
    limit = limit * 1.05

    # choose to draw wireframe box or square at z=0 based on `shape` argument
    if shape == 'box':
        # create wireframe box
        x = [-limit, limit]
        y = [-limit, limit]
        z = [-limit, limit]
        for s, e in combinations(np.array(list(product(x, y, z))), 2):
            if np.sum(np.abs(s - e)) == x[1] - x[0]:
                mlab.plot3d(*zip(s, e), color=color, tube_radius=tube_radius, figure=figure)
    elif shape == 'square':
        # create square at z=0
        mlab.plot3d([limit, limit, -limit, -limit, limit],
                    [limit, -limit, -limit, limit, limit],
                    [offset, offset, offset, offset, offset],
                    color=color,
                    opacity=opacity,
                    tube_radius=tube_radius,
                    figure=figure)
    else:
        raise ValueError(f"Unknown shape: {shape}. Choose either 'box' or 'square'.")

    if view := kwargs.get('view', {'azimuth': 180, 'elevation': 90}):
        mlab.view(**view)

    if (filename := kwargs.get('filename', '3D_contour.png')) and kwargs.get('save', True):
        mlab.savefig(filename)

    if kwargs.get('show', False):
        mlab.show()

    return figure
