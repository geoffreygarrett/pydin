def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def ellipse_of_intersection(l, m, n, f, a, b, c):
    # Establish normal vector
    normal = np.array([l, m, n])
    k = np.linalg.norm(normal)

    # Divide the equation by κ = √l2 + m2 + n2 if f ≥ 0, and otherwise divide by −κ.
    normal_scaled = normal / k if f >= 0 else -normal / k
    h = f / k if f >= 0 else -f / k

    # Reorder components such that the largest absolute value is in the last place
    # Divide the equation (39) by κ = √l2 + m2 + n2 if f ≥ 0, and otherwise
    # divide (39) by −κ. Then the plane (39) is equally specified by the scaled equation
    max_idx = np.argsort(np.abs(normal_scaled))[-1]
    indices = np.array([0, 1, 2])
    indices[-1], indices[max_idx] = indices[max_idx], indices[-1]
    print(f"indices[0]: {indices[0]}")
    print(f"indices[1]: {indices[1]}")
    print(f"indices[2]: {indices[2]}")
    # ONLY place tyhe largest absolute value in the last place
    # indices = np.array([0, 1, 2])
    # max_idx = np.argmax(np.abs(normal_scaled))
    # indices[-1], indices[max_idx] = indices[max_idx], indices[-1]
    # print(indices)

    # Create a permutation matrix to keep track of the transformation
    P = np.zeros((3, 3))
    P[np.arange(3), indices] = 1

    # Apply permutation to the scaled normal vector
    λ, μ, v = P @ normal_scaled

    # Apply permutation to the semi-axis lengths
    sx, sy, sz = P @ np.array([a, b, c])

    # Calculate the ellipse coefficients
    A = v ** 2 * sz ** 2 / sx ** 2 + λ ** 2
    B = 2 * λ * μ
    C = v ** 2 * sz ** 2 / sy ** 2 + μ ** 2
    D = -2 * h * λ
    E = -2 * h * μ
    G = h ** 2 - v ** 2 * sz ** 2

    # print("Ellipse coefficients: ")
    # print(f"A = {A}")
    # print(f"B = {B}")
    # print(f"C = {C}")
    # print(f"D = {D}")
    # print(f"E = {E}")
    # print(f"G = {G}")

    # A * x^2 + B * x * y + C * y^2 + D * x + E * y + G = 0
    # m0 = np.array([
    #     [G, D / 2, E / 2],
    #     [D / 2, A, B / 2],
    #     [E / 2, B / 2, C]
    # ])
    M = np.array([
        [A, B / 2],
        [B / 2, C]
    ])
    q = np.array([D, E])
    w = -G

    # Print all values for verification
    print(f"{A=},\n{B=},\n{C=},\n{D=},\n{E=},\n{G=}")

    # Get eigenvalues and eigenvectors
    # eigvals, eigvecs = np.linalg.eig(m1)
    # lambda1, lambda2 = sorted(eigvals)
    # eigvec1, eigvec2 = eigvecs[:, eigvals.argmin()], eigvecs[:, eigvals.argmax()]

    # Calculate the eigenvalues and eigenvectors of M (Eq. 120)
    eig1 = 0.5 * (A + C + np.sqrt((A - C) ** 2 + B ** 2))
    eig2 = 0.5 * (A + C - np.sqrt((A - C) ** 2 + B ** 2))
    eigv1 = np.array([B, 2 * (A - eig1)])
    eigv2 = np.array([B, 2 * (A - eig2)])
    eigv2_norm = np.linalg.norm(eigv1)
    eigv2_normalised = eigv1 / eigv2_norm if eigv2_norm != 0 else eigv2
    print(f"{eig1=},\n{eig2=},\n{eigv1=} \n{eigv2=}")

    # Get theta from the eigenvector corresponding to the largest eigenvalue
    # theta = np.arccos(np.dot(eigv2_normalised, np.array([1, 0])))
    theta = np.arctan2(B, A - C) / 2

    # det_m = A * C - B ** 2 / 4

    # The ellipse is centred at u, where 2Mu = −q
    u = -0.5 * np.linalg.inv(M) @ q
    const = w + u.T @ M @ u

    # a0 = calc_a(lambda1, det_m0, det_m)
    # b0 = calc_b(lambda2, det_m0, det_m)
    # Calculate the parameters a0 and b0 (123)
    d1 = a0 = np.sqrt(const / eig2)
    d2 = b0 = np.sqrt(const / eig1)

    print(f"{a0=},\n{b0=},\n{theta=}\n")

    # Calculate the center of the ellipse
    x0, y0 = u
    z0 = (h - λ * x0 - μ * y0) / v
    # print(f"{x0=},\n{y0=},\n{z0=}\n")

    A_hat = A
    B_hat = B * v
    C_hat = C * v ** 2
    D_hat = D
    E_hat = E * v

    # _f = f +

    # Calculate the d_hat1 and d_hat2
    # eig1_hat = 0.5 * (A_hat + C_hat + np.sqrt((A_hat - C_hat) ** 2 + B_hat ** 2))
    # eig2_hat = 0.5 * (A_hat + C_hat - np.sqrt((A_hat - C_hat) ** 2 + B_hat ** 2))
    #
    # d_hat1 = np.sqrt(f / eig2_hat)
    # d_hat2 = np.sqrt(f / eig1_hat)

    # Update the center to be in the original coordinates
    center_in_new_coords = np.array([x0, y0, z0])
    center_in_old_coords = P.T @ center_in_new_coords  # Transpose back to rotate into old coordinates

    print(f"Center in new coordinates: {center_in_new_coords}")
    print(f"Center in old coordinates: {center_in_old_coords}")

    x0, y0, z0 = center_in_old_coords
    # x0=1.4705882352941178,
    # y0=0.5294117647058822,
    # z0=1.26293583244616e-309,
    # Calculate the angles theta, phi and omega
    qx = λ
    qy = μ
    qz = v

    print(f"{qx=},\n{qy=},\n{qz=}\n")

    # theta = np.arctan2(normalized_b, normalized_a - normalized_c) / 2
    phi = np.pi / 2 - np.arctan2(np.abs(qz), np.sqrt(qx ** 2 + qy ** 2))
    omega = np.pi - np.arctan2(qy, qx)

    # Aη2 + (Bν)ηζ + (Cν2)ζ2 + Dη + (Eν)ζ = w, (Eq. 125)
    #
    # print(f"{np.rad2deg(phi)=},\n{np.rad2deg(omega)=}\n")

    # Create transformation matrix
    t = np.eye(4)
    R = P.T @ rot3(-omega) @ rot2(-phi)
    eigv2_normalised_3d = np.array([eigv2_normalised[0], eigv2_normalised[1], 0])

    uz = np.array([λ, μ, v])
    ux = np.array([1, 0, 0]) - np.dot(np.array([1, 0, 0]), uz) * uz
    uy = np.cross(uz, ux)
    uy = uy / np.linalg.norm(uy)

    return (a0, b0, ai, bi), (x0, y0, z0), t, t_inv, theta, theta_inv, pole


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


def plot_plane(l, m, n, f, limit, resolution=10, color=(0, 0, 1), opacity=0.7):
    # Generate grid of points
    u = np.linspace(-limit, limit, resolution)
    v = np.linspace(-limit, limit, resolution)

    U, V = np.meshgrid(u, v)

    # Plane equation: l * x + m * y + n * z + f = 0
    if n != 0:
        # Solve equation for Z if possible
        X = U
        Y = V
        Z = (f - l * X - m * Y) / n
    elif m != 0:
        # If not, solve for Y
        X = U
        Z = V
        Y = (f - l * X - n * Z) / m
    elif l != 0:
        # If not, solve for X
        Y = U
        Z = V
        X = (f - m * Y - n * Z) / l
    else:
        raise ValueError("At least one of l, m, n must be nonzero")

    # Plot the surface with wireframe
    mlab.mesh(X, Y, Z, color=color, opacity=opacity / 2)
    mlab.mesh(X, Y, Z, color=color, representation='wireframe', opacity=opacity)

    return mlab


def calculate_polar_plane_pole(a, b, c, l, m, n, f):
    if np.abs(f) < 1e-10:
        px = a * a * l
        py = b * b * m
        pz = c * c * n
    else:
        px = a * a * l / f
        py = b * b * m / f
        pz = c * c * n / f
    return np.array([px, py, pz])


def intersects_ellipse_interior(m, r, s, A, B, p0, p1):
    # m = center of 3d ellipse
    # r, s = vectors perpendicular to normal vector of plane
    # A, B = semi-major and semi-minor axes of ellipse
    # p0, p1 = endpoints of line segment
    n = np.cross(r, s)
    # get plane offset
    d = -np.dot(n, m)
    # get intersection of line and plane
    t = -(np.dot(n, p0) + d) / (np.dot(n, p1 - p0))
    # get point on plane
    q = p0 + t * (p1 - p0)
    # check if point is inside ellipse
    return (q[0] - m[0]) ** 2 / A ** 2 + (q[1] - m[1]) ** 2 / B ** 2 <= 1


def intersects_ellipse_interior(m, r, s, A, B, p0, p1):
    # m = center of 3d ellipse
    # r, s = vectors perpendicular to normal vector of plane
    # A, B = semi-major and semi-minor axes of ellipse
    # p0, p1 = endpoints of line segment

    # calculate the normal of the plane the ellipse lies in
    n = np.cross(r, s)

    # get plane offset
    d = -np.dot(n, m)

    # calculate t for multiple points in p1
    t = -(np.dot(n, p0) + d) / np.einsum('i,ijkl->jkl', n, (p1 - np.reshape(p0, (3, 1, 1, 1))))

    # get points on plane for multiple points in p1
    q = p0[:, np.newaxis, np.newaxis, np.newaxis] + t * (p1 - p0[:, np.newaxis, np.newaxis, np.newaxis])

    # get displacement from the center of the ellipse
    displacement = q - m[:, np.newaxis, np.newaxis, np.newaxis]

    # project displacement onto r and s, then divide by A and B, respectively
    r_component = np.einsum('i,ijkl->jkl', r, displacement) / A
    s_component = np.einsum('i,ijkl->jkl', s, displacement) / B

    # check if points are inside ellipse
    return r_component ** 2 + s_component ** 2 <= 1


def other_sol(
        n,  # normal vector
        f,  # distance from origin
        a, b, c  # ellipsoid parameters
):
    # n : unit normal vector of plane
    # q : arbitrary point on plane, interior to the ellipse
    # r, s : arbitrary vectors perpendicular to n
    # D1 : diagonal matrix with 1/a, 1/b, 1/c
    # t : first parameter of the 2D parametric equation of the ellipse
    # u : second parameter of the 2D parametric equation of the ellipse
    # t0 : first parameter of the 2D parametric equation of the ellipse, center of ellipse
    # u0 : second parameter of the 2D parametric equation of the ellipse, center of ellipse
    # A : semi-major axis of ellipse
    # B : semi-minor axis of ellipse
    # k : distance of the plane from the origin
    # Define D1 as a diagonal matrix with 1/a, 1/b, 1/c
    D1 = np.diag([1 / a, 1 / b, 1 / c])

    # Define any point on plane, interior to the ellipse
    q = n * f / np.linalg.norm(n)

    # Choose r and s arbitrarily for
    # satisfying (r, r) = (s, s) = 1, (n, r) = (n, s) = 0, (r, s) = 0
    # find an arbitrary vector r perpendicular to n
    for e in np.eye(n.size):
        r = np.cross(e, n)
        if np.any(r):
            break

    r = r - np.dot(r, n) * n
    s = np.cross(n, r)

    # In case (7) is not fulfilled for the chosen r and s, dot(D1 @ r, D1 @ s) != 0, we transform:
    if np.dot(D1 @ r, D1 @ s) != 0:
        if np.dot(D1 @ r, D1 @ r) - np.dot(D1 @ s, D1 @ s) == 0:
            omega = np.pi / 4
        else:
            omega = 0.5 * np.arctan(2 * np.dot(D1 @ r, D1 @ s) / (np.dot(D1 @ r, D1 @ r) - np.dot(D1 @ s, D1 @ s)))

        r = np.cos(omega) * r + np.sin(omega) * s
        r = r / np.linalg.norm(r)
        r = r - np.dot(r, n) * n
        # s = -np.sin(omega) * r + np.cos(omega) * s
        s = np.cross(n, r)
        s = s / np.linalg.norm(s)

        print(f"(r,r) = {np.dot(r, r)}")
        print(f"(s,s) = {np.dot(s, s)}")
        print(f"(r,s) = {np.dot(r, s)}")
        print(f"(n,r) = {np.dot(n, r)}")
        print(f"(n,s) = {np.dot(n, s)}")

    # Calculate beta_1 and beta_2
    beta_1 = np.dot(D1 @ r, D1 @ r)
    beta_2 = np.dot(D1 @ s, D1 @ s)
    print(f"beta_1 = {beta_1}")
    print(f"beta_2 = {beta_2}")
    print(f"beta_1 * beta_2 = {beta_1 * beta_2}")
    print(f"(n,n) = {np.dot(n, n)}")
    print(f"(r,r) = {np.dot(r, r)}")
    print(f"(s,s) = {np.dot(s, s)}")
    print(f"(r,s) = {np.dot(r, s)}")
    print(f"(n,r) = {np.dot(n, r)}")
    print(f"(n,s) = {np.dot(n, s)}")
    print(f"CHECK: {np.dot(D1 @ r, D1 @ r)=}")
    print(f"CHECK: {np.dot(D1 @ s, D1 @ s)=}")
    print(f"CHECK: {np.dot(D1 @ r, D1 @ s)=}")
    print(f"CHECK: {np.dot(D1 @ r, D1 @ r) - np.dot(D1 @ s, D1 @ s)=}")
    print(f"CHECK: {np.dot(D1 @ r, D1 @ s)}")

    # Expression for d (Eq. 11)
    k = np.dot(q, n)
    d = k ** 2 / (a ** 2 * n[0] ** 2 + b ** 2 * n[1] ** 2 + c ** 2 * n[2] ** 2)

    # Test
    D1_tilde = np.diag([1 / b / c, 1 / a / c, 1 / a / b])
    print(np.dot(D1_tilde @ n, D1_tilde @ n))

    assert beta_1 * beta_2 - np.dot(D1_tilde @ n, D1_tilde @ n) <= 1e-10

    # Ellipse center, m (Eq. 39)
    print(beta_1, beta_2)
    print(k)
    if k == 0:
        m = np.zeros(3)
    else:
        m = k * (n - np.dot(D1 @ n, D1 @ r) / beta_1 * r - np.dot(D1 @ n, D1 @ s) / beta_2 * s)

    # Semi-axes, A and B (Eq. 10)
    A = np.sqrt((1 - d) / beta_1)
    B = np.sqrt((1 - d) / beta_2)

    # Ellipse center in 2D
    t0 = - np.dot(D1 @ q, D1 @ r) / np.dot(D1 @ r, D1 @ r)
    u0 = - np.dot(D1 @ q, D1 @ s) / np.dot(D1 @ s, D1 @ s)

    return m, r, s, A, B


def polar_plane_from_point(p, a, b, c):
    f = 1.0 / np.sqrt((p[0] / a ** 2) ** 2 + (p[1] / b ** 2) ** 2 + (p[2] / c ** 2) ** 2)
    l = p[0] * f / a ** 2
    m = p[1] * f / b ** 2
    n = p[2] * f / c ** 2
    return l, m, n, f


if __name__ == '__main__':
    import numpy as np
    from mayavi import mlab
    from pydin import plotting
    from pydin.core import rot2, rot3
    from pydin.core.shape import calculate_polar_plane_pole, ellipse_of_intersection, marching_cubes
    from pydin import logging

    logging.set_level(logging.DEBUG)
    assert plotting is not None

    figure = mlab.figure('3D Contour', bgcolor=(0.2, 0.2, 0.2), fgcolor=(1., 1., 1.), size=(1000, 800))

    a = 3.
    b = 2.
    c = 1.


    def voxel_carving(p_list):
        n_grid = 120j
        lim_factor = 1.5
        a_lim = lim_factor * a
        b_lim = lim_factor * b
        c_lim = lim_factor * c
        grid = np.mgrid[-a_lim:a_lim:n_grid, -b_lim:b_lim:n_grid, -c_lim:c_lim:n_grid]
        active_grid = np.ones(grid.shape[1:], dtype=bool)

        for p in p_list:
            l, m, n, f = polar_plane_from_point(p, a, b, c)
            m_ellipse, r_ellipse, s, A, B = ellipse_of_intersection(l, m, n, f, a, b, c)
            theta_sample = np.linspace(0, 2 * np.pi, 100)

            # Expand dimensions of theta_cos and theta_sin to make them (100, 1)
            xyz = m_ellipse + np.outer(A * np.cos(theta_sample), r_ellipse) + np.outer(B * np.sin(theta_sample), s)
            mlab.plot3d(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=(1, 1, 1), tube_radius=0.04)
            pole = calculate_polar_plane_pole(l, m, n, f, a, b, c)
            mlab.points3d(*pole, color=(1, 1, 1), scale_factor=0.4)

            ######

            # l = 0  # x component of normal vector
            # m = 0  # y component of normal vector
            # n = 1  # z component of normal vector
            # f = 7.9  # distance from origin

            visual_hull = intersects_ellipse_interior(m_ellipse, r_ellipse, s, A, B, pole, grid)
            active_grid = np.logical_and(active_grid, visual_hull)

            # plot cone from polar point to the x1,xy,xz
            X = np.array([pole, m_ellipse])
            mlab.plot3d(X[:, 0], X[:, 1], X[:, 2], color=(1, 1, 1), tube_radius=0.04)

            Xdif = xyz[:, 0] - pole[0]
            Ydif = xyz[:, 1] - pole[1]
            Zdif = xyz[:, 2] - pole[2]

            mlab.quiver3d(pole[0] * np.ones_like(Xdif), pole[1] * np.ones_like(Ydif), pole[2] * np.ones_like(Zdif),
                          Xdif, Ydif,
                          Zdif, color=(0.7, 0.7, 1), scale_factor=1, mode='2ddash', opacity=0.1)

        return grid, active_grid


    grid, grid_inside_mask = voxel_carving([
        # np.array([0, 7, 7]),
        np.array([0, -7, 7]),
        np.array([0, 7, -7]),
        np.array([7, 0, 0]),
        np.array([0, -7, -7]),
        np.array([0, 7, 0]),
        np.array([0, -7, 0]),

    ])
    # grid, grid_inside_mask = voxel_carving([np.array([0, -2, 2]), np.array([0, 2, -2])])

    # grid_inside_mask = np.where(grid[0] ** 2 / a ** 2 + grid[1] ** 2 / b ** 2 + grid[2] ** 2 / c ** 2 <= 1, 1, 0)
    grid_inside_mask = np.where(grid_inside_mask, 1, 0)
    mesh = marching_cubes(grid, grid_inside_mask, 0.1)

    faces = np.array(mesh.get_facets())
    vertices = np.array(mesh.get_vertices())
    # print(faces)
    # print(vertices)

    # Separate the vertices into x, y, and z components
    # print(vertices)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    # print(x, y, z)

    # Create the triangular mesh wireframe
    mlab.triangular_mesh(x, y, z, faces,
                         # wireframe setting:
                         representation='wireframe',
                         color=(0.2, 0.2, 1),
                         opacity=0.2,
                         line_width=0.1,
                         figure=figure)

    mlab.triangular_mesh(x, y, z, faces,
                         # wireframe setting:
                         representation='surface',
                         color=(0.2, 0.2, 1),
                         opacity=0.1,
                         line_width=0.1,
                         figure=figure)


    # mlab.triangular_mesh(x, y, z, faces, color=(0.5, 0.5, 0.5), figure=figure, opacity=0.6)
    # mlab.points3d(x, y, z, scale_factor=0.015, color=(1, 1, 1), figure=figure)

    # print(grid)
    # print(grid_inside_mask.shape)
    # plot grid inside
    # mlab.points3d(grid[0][grid_inside_mask == 1], grid[1][grid_inside_mask == 1], grid[2][grid_inside_mask == 1],
    #               scale_factor=0.02, color=(1, 1, 1), figure=figure, opacity=0.5)

    ######

    def sample_ellipse(a, b, theta):
        t = np.linspace(0, 2 * np.pi, 100)
        x = a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
        y = b * np.sin(t) * np.cos(theta) + a * np.cos(t) * np.sin(theta)
        z = np.zeros_like(t)
        return x, y, z


    def plot_ellipse(a, b, x0, y0, z0, theta, R):
        x, y, z = sample_ellipse(a, b, theta)
        mlab.points3d(x0, y0, z0, color=(1, 1, 1), scale_factor=0.3)
        x, y, z = np.dot(R, np.array([x, y, z]))
        x = x + x0
        y = y + y0
        z = z + z0
        mlab.plot3d(x, y, z, color=(1, 1, 1))


    def plot_projection_lines(a0, b0, x0, y0, z0, theta0, ai, bi, theta_i, R1, R2):
        x1, y1, z1 = sample_ellipse(a0, b0, theta0)
        x1, y1, z1 = np.dot(R1, np.array([x1, y1, z1]))
        x1 += x0
        y1 += y0
        z1 += 0
        x2, y2, z2 = sample_ellipse(ai, bi, theta_i)
        x2, y2, z2 = np.dot(R2, np.array([x2, y2, z2]))
        x2 += x0
        y2 += y0
        z2 += z0
        mlab.quiver3d(x2, y2, z2, x1 - x2, y1 - y2, z1 - z2, color=(1, 1, 1), scale_factor=1)


    # plot_ellipsoid_mayavi(a, b, c, figure=figure, color=(1, 0, 0), opacity=0.2)

    mlab.show()
    raise SystemExit
