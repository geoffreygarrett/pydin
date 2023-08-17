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
from pydin.core import rot3, rot2, rot1


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

    # # Normalize the input normal vector to get the new z-axis
    # uz = normal_scaled
    #
    # # Choose a random vector not parallel to uz to compute the cross product
    # if uz[0] == 1 and uz[1] == 0 and uz[2] == 0:
    #     temp = np.array([0, 1, 0])
    # else:
    #     temp = np.array([1, 0, 0])
    #
    # # Compute cross product to find the new x-axis
    # ux = np.cross(temp, uz)
    # ux = ux / np.linalg.norm(ux)  # normalize
    #
    # # Compute cross product to find the new y-axis
    # uy = np.cross(uz, ux)
    # uy = uy / np.linalg.norm(uy)  # normalize
    #
    # # Stack the new axes together to form the rotation matrix
    # R = np.array([ux, uy, uz])

    # R = P.T @ R.T Z
    # R = P.T #@ rot1(v)

    # @ np.array([ux, uy, uz]).T
    # R = P.T @ np.array([ux, uy, uz]).T
    t=np.eye(4)
    t[:3, :3] = R
    t[:3, 3] = np.array([x0, y0, z0])

    t_inv = np.linalg.inv(t)
    x0_t, y0_t, z0_t = t_inv[:3, 3]
    # A_inv, B_inv, C_inv, D_inv, E_inv, F_inv = calculate_terms(x0_t, y0_t, z0_t, sx, sy, sz, psi_t, eps_t, omega_t)
    A_inv = A_hat
    B_inv = B_hat
    C_inv = C_hat
    D_inv = D_hat
    E_inv = E_hat
    F_inv = G

    print(f"{A_inv=},\n{B_inv=},\n{C_inv=},\n{D_inv=},\n{E_inv=},\n{F_inv=}")

    # M0_inv = np.array([
    #     [F_inv, D_inv / 2, E_inv / 2],
    #     [D_inv / 2, A_inv, B_inv / 2],
    #     [E_inv / 2, B_inv / 2, C_inv]
    # ])

    M_inv = np.array([
        [A_inv, B_inv / 2],
        [B_inv / 2, C_inv]
    ])

    # The ellipse is centred at u, where 2Mu = −q
    q_inv = np.array([D_inv, E_inv])
    w_inv = -F_inv
    u = -0.5 * np.linalg.inv(M_inv) @ q_inv
    const = w + u.T @ M @ u

    # Calculate the eigenvalues and eigenvectors of M (Eq. 120)
    eig1 = 0.5 * (A_inv + C_inv + np.sqrt((A_inv - C_inv) ** 2 + B_inv ** 2))
    eig2 = 0.5 * (A_inv + C_inv - np.sqrt((A_inv - C_inv) ** 2 + B_inv ** 2))
    eigv1 = np.array([B_inv, 2 * (A_inv - eig1)])
    eigv2 = np.array([B_inv, 2 * (A_inv - eig2)])
    # eigv2_norm = np.linalg.norm(eigv2)
    # eigv2_normalised = eigv2 / eigv2_norm if eigv2_norm != 0 else eigv2
    # print(f"{eig1=},\n{eig2=},\n{eigv1=} \n{eigv2=}")

    eigvals, eigvecs = np.linalg.eig(M)
    lambda1, lambda2 = sorted(eigvals)
    print(f"{lambda1=},\n{lambda2=}")
    print(f"{eigvecs=}")
    # det_m0 = np.linalg.det(M0_inv)
    # det_m = np.linalg.det(M)
    # ai = calc_a(lambda1, det_m0, det_m)
    # bi = calc_b(lambda2, det_m0, det_m)
    # Calculate the parameters a0 and b0 (123)
    d1 = ai = np.sqrt(const / eig2)
    d2 = bi = np.sqrt(const / eig1)

    # Get eigenvalues and eigenvectors
    # eigvals_inv, eigvecs_inv = np.linalg.eig(m_inv)
    # lambda1_inv, lambda2_inv = sorted(eigvals_inv)

    # Calculate the parameters ai and bi
    # det_m0_inv = np.linalg.det(m0_inv)
    # det_m_inv = np.linalg.det(m_inv)
    # ai = calc_a(lambda1_inv, det_m0_inv, det_m_inv)
    # bi = calc_b(lambda2_inv, det_m0_inv, det_m_inv)
    # ai = d_hat1
    # bi = d_hat2
    theta_inv = np.arctan2(B_hat, A_hat - C_hat) / 2
    # theta_inv = np.arccos(np.dot(eigv2_normalised, np.array([1, 0])))
    print(f"{ai=},\n{bi=},\n{theta_inv=}\n")

    # Pole of the plane
    if np.abs(f) < 1e-10:
        px = sx ** 2 * λ
        py = sy ** 2 * μ
        pz = sz ** 2 * v
        pole = P.T @ np.array([px, py, pz])
    else:
        px = a ** 2 * l / f
        py = b ** 2 * m / f
        pz = c ** 2 * n / f
        pole = np.array([px, py, pz])

    # px = a ** 2 * l / f
    # py = b ** 2 * m / f
    # pz = c ** 2 * n / f

    # Or
    # px = sx ** 2 * λ / h
    # py = sy ** 2 * μ / h
    # pz = sz ** 2 * v / h

    # pole = P @ pole

    # t[:3, :3] = R @ rot3(-theta_inv)
    # t_inv = np.linalg.inv(t)

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


def plot_plane(l, m, n, f, limit, resolution=10):
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

    # Plot the surface
    mlab.mesh(X, Y, Z, color=(0, 0, 1), opacity=0.2)

    return mlab


if __name__ == '__main__':
    import numpy as np
    from mayavi import mlab
    from pydin import plotting
    from pydin.core.shape import ellipse_of_intersection as ellipse_of_intersection2

    assert plotting is not None

    figure = mlab.figure('3D Contour', bgcolor=(0.2, 0.2, 0.2), fgcolor=(1., 1., 1.), size=(1000, 800))
    a = 10.
    b = 2.
    c = 3.

    epsilon = 0
    l = 0.5
    m = 1
    n = 0
    f = 1
    l += epsilon
    m += epsilon
    n += epsilon

    A, B, C, D, E, G, aa0, bb0, aai, bbi, theta_t, xx0, yy0, zz0 = ellipse_of_intersection2(l, m, n, f, a, b, c)
    print(
        f"A={A},\nB={B},\nC={C},\nD={D},\nE={E},\nG={G},\na0={aa0},\nb0={bb0},\nx0={xx0},\ny0={yy0},\nz0={zz0},\nai={aai},\nbi={bbi}")

    (a0, b0, ai, bi), (x0, y0, z0), T, T_inv, theta, theta_inv, pole = ellipse_of_intersection(l, m, n, f, a, b, c)
    a0 = aa0
    b0 = bb0
    ai = aai
    bi = bbi
    x0 = xx0
    y0 = yy0
    z0 = zz0
    # x0=-0.3824091778202678,
    # y0=-0.13766730401529645,
    # z0=2.5200764818355643,
    print(f"{x0=},\n{y0=},\n{z0=}")
    # x0=1.4705882352941178,
    # y0=0.5294117647058822,
    # z0=1.26293583244616e-309,
    # plot_ellipsoid_mayavi(a, b, c, figure=figure, color=(1, 0, 0), opacity=0.8)
    limit = np.max([a, b, c]) * 2
    t = np.linspace(0, 2 * np.pi, 100)
    x = ai * np.cos(t) * np.cos(theta_t) - bi * np.sin(t) * np.sin(theta_t)
    y = bi * np.sin(t) * np.cos(theta_t) + ai * np.cos(t) * np.sin(theta_t)
    # rotate x and y by theta_inv
    z = np.zeros_like(t)
    # x, y, z = np.dot(rot3(-theta_t), np.array([x, y, z]))

    # # # Apply rotation using broadcasting
    # x, y, z = np.dot(rot3(-theta), np.array([x, y, z]))

    # Combine x, y, z into a 3xN array
    points = np.array([x, y, z])

    # Apply rotation using broadcasting
    rotated_points = np.dot(T[:3, :3], points)
    rotated_points = rotated_points + np.array([[x0], [y0], [z0]])

    # Extract the rotated x, y, z coordinates
    x_rot, y_rot, z_rot = rotated_points

    mlab.points3d(x0, y0, z0, color=(1, 1, 1), scale_factor=0.5)
    mlab.points3d(*pole, color=(1, 1, 1), scale_factor=0.5)

    mlab.plot3d(x_rot, y_rot, z_rot, color=(1, 1, 1))
    # mlab.plot3d(x, y, z, color=(1, 0, 0))
    # plot plane Ax + By + Cz + D = 0
    # Define ranges for x and y

    # t = np.linspace(0, 2 * np.pi, 100)
    # x = a0 * np.cos(t)
    # y = b0 * np.sin(t)
    # # rotate x and y by theta_inv
    # z = np.zeros_like(t)
    # x, y, z = np.dot(rot3(-theta), np.array([x, y, z]))
    # mlab.plot3d(x, y, z, color=(1, 0, 0))

    plot_plane(l, m, n, f, limit, resolution=10)
    plot_plane(0, 0, 1, 0, limit, resolution=10)

    # u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    # x = a * np.cos(u) * np.sin(v)
    # y = b * np.sin(u) * np.sin(v)
    # z = c * np.cos(v)
    # mlab.mesh(x, y, z, color=(0, 1, 0), opacity=0.2)
    plot_ellipsoid_mayavi(a, b, c, figure=figure, color=(1, 0, 0), opacity=0.8)

    mlab.show()
    raise SystemExit
