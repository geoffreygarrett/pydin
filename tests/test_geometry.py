from pydin.core import rot2, rot3


def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def ellipse_of_intersection(ax, ay, az, ad, a, b, c):
    # Normalize ax + by + cz + d = 0
    ax, ay, az, ad = normalize_vector([ax, ay, az, ad])

    normalized_a = 1 / a ** 2 + ax ** 2 / (az ** 2 * c ** 2)
    normalized_b = 2 * ax * ay / (az ** 2 * c ** 2)
    normalized_c = 1 / b ** 2 + ay ** 2 / (az ** 2 * c ** 2)
    normalized_d = 2 * ax * ad / (az ** 2 * c ** 2)
    normalized_e = 2 * ay * ad / (az ** 2 * c ** 2)
    normalized_f = ad ** 2 / (az ** 2 * c ** 2) - 1

    m0 = np.array([
        [normalized_f, normalized_d / 2, normalized_e / 2],
        [normalized_d / 2, normalized_a, normalized_b / 2],
        [normalized_e / 2, normalized_b / 2, normalized_c]
    ])

    m = np.array([
        [normalized_a, normalized_b / 2],
        [normalized_b / 2, normalized_c]
    ])

    # Get eigenvalues and eigenvectors
    eigvals, _ = np.linalg.eig(m)
    lambda1, lambda2 = sorted(eigvals)

    # Calculate the parameters a0 and b0
    det_m0 = np.linalg.det(m0)
    det_m = np.linalg.det(m)
    a0 = np.sqrt(-det_m0 / (det_m * lambda1))
    b0 = np.sqrt(-det_m0 / (det_m * lambda2))

    # Calculate the center of the ellipse
    denominator = 4 * normalized_a * normalized_c - normalized_b ** 2
    x0 = (normalized_b * normalized_e - 2 * normalized_c * normalized_d) / denominator
    y0 = (normalized_b * normalized_d - 2 * normalized_a * normalized_e) / denominator
    z0 = (ax * x0 + ay * y0 + ad) / az

    # Calculate the angles theta, phi and omega
    qx = ax * ad / np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    qy = ay * ad / np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    qz = az * ad / np.sqrt(ax ** 2 + ay ** 2 + az ** 2)

    theta = np.arctan2(normalized_b, normalized_a - normalized_c) / 2
    phi = np.pi / 2 - np.arctan2(np.abs(qz), np.sqrt(qx ** 2 + qy ** 2))
    omega = np.pi - np.arctan2(qy, qx)

    # Create transformation matrix
    t = np.eye(4)
    t[:3, :3] = rot3(-omega) @ rot2(-phi) @ rot3(theta)
    t[:3, 3] = np.array([x0, y0, z0])

    return (a0, b0), (x0, y0, z0), t, theta


import numpy as np


def test_ellipse_of_intersection():
    a, b, c = 7., 5., 4.
    ax, ay, az, ad = 1., 1., 1., 1.

    (a0, b0), (x0, y0, z0), t, theta = ellipse_of_intersection(ax, ay, az, ad, a, b, c)

    # Check semi-axes of the intersection ellipse
    assert a0 > 0
    assert b0 > 0

    # Check center of the intersection ellipse
    assert -a <= x0 <= a
    assert -b <= y0 <= b
    assert -c <= z0 <= c

    # Check orientation angle of the intersection ellipse
    assert -np.pi <= theta <= np.pi


def test_ellipse_of_intersection_with_horizontal_plane():
    a, b, c = 7., 5., 4.
    ax, ay, az, ad = 0., 0., 1., 0.

    (a0, b0), (x0, y0, z0), t, theta = ellipse_of_intersection(ax, ay, az, ad, a, b, c)

    # The intersection ellipse should be the same as the cross section of the ellipsoid
    assert np.isclose(a0, a)
    assert np.isclose(b0, b)
    assert np.isclose(x0, 0.)
    assert np.isclose(y0, 0.)
    assert np.isclose(z0, 0.)


def test_ellipse_of_intersection_with_ellipsoid_center():
    a, b, c = 7., 5., 4.
    ax, ay, az, ad = 0., 0., 1., -0.5

    (a0, b0), (x0, y0, z0), t, theta = ellipse_of_intersection(ax, ay, az, ad, a, b, c)

    # The center of the intersection ellipse should be at the center of the ellipsoid
    assert np.isclose(x0, 0.)
    assert np.isclose(y0, 0.)
    assert np.isclose(z0, -0.5)


# Test if the function returns the correct ellipse when the plane intersects the center of a sphere
def test_ellipse_of_intersection_with_sphere_center():
    a, b, c = 1., 1., 1.  # sphere with radius 1
    ax, ay, az, ad = 0., 0., 1., 0.  # plane through the origin

    (a0, b0), (x0, y0, z0), t, theta = ellipse_of_intersection(ax, ay, az, ad, a, b, c)

    assert np.isclose(a0, 1.)
    assert np.isclose(b0, 1.)
    assert np.isclose(x0, 0.)
    assert np.isclose(y0, 0.)
    assert np.isclose(z0, 0.)


# Test if the function returns the correct ellipse when the plane intersects a non-central point of a sphere
def test_ellipse_of_intersection_with_sphere_non_center():
    a, b, c = 2., 2., 2.  # sphere with radius 2
    ax, ay, az, ad = 0., 0., 1., -1.  # plane intersects 1 unit below the center of the sphere

    (a0, b0), (x0, y0, z0), t, theta = ellipse_of_intersection(ax, ay, az, ad, a, b, c)

    assert np.isclose(a0, np.sqrt(3))  # expected radius of the intersection circle
    assert np.isclose(b0, np.sqrt(3))
    assert np.isclose(x0, 0.)
    assert np.isclose(y0, 0.)
    assert np.isclose(z0, -1.)


# Test if the function returns the correct ellipse when the plane is tangent to a sphere
def test_ellipse_of_intersection_with_sphere_tangent():
    a, b, c = 3., 3., 3.  # sphere with radius 3
    ax, ay, az, ad = 0., 0., 1., -3.  # plane is tangent to the sphere at the bottom

    (a0, b0), (x0, y0, z0), t, theta = ellipse_of_intersection(ax, ay, az, ad, a, b, c)

    assert np.isclose(a0, 0., atol=1e-6)  # intersection is a single point
    assert np.isclose(b0, 0., atol=1e-6)
    assert np.isclose(x0, 0.)
    assert np.isclose(y0, 0.)
    assert np.isclose(z0, -3.)


def test_transformation_matrix_correctness():
    # Define an ellipsoid and a plane
    a, b, c = 3., 5., 7.
    ax, ay, az, ad = 0., 0., 1., -5.

    # Get the ellipse of intersection and the transformation matrix
    (a0, b0), (x0, y0, z0), t, theta = ellipse_of_intersection(ax, ay, az, ad, a, b, c)

    # Generate points on the ellipse in the 2D plane
    thetas = np.linspace(0, 2 * np.pi, 100)
    ellipse_points_2d = np.array([a0 * np.cos(thetas), b0 * np.sin(thetas), np.ones_like(thetas)]).T

    # Add a fourth dimension to the 2D ellipse points (to make it Nx4)
    ellipse_points_2d = np.column_stack([ellipse_points_2d, np.ones(ellipse_points_2d.shape[0])])

    # Use the transformation matrix to move the points into 3D space
    ellipse_points_3d = (t @ ellipse_points_2d.T).T

    # Check that each point is on the plane and the ellipsoid
    for x, y, z, _ in ellipse_points_3d:
        plane_c = (ax * x + ay * y + az * z - ad)

        print(plane_c)

        # Check if the point is on the plane (with a tolerance due to possible floating point errors)
        # assert np.isclose(plane_c, 0, atol=1e-6)

        # Calculate C
        ellipse_c = (x ** 2 / a ** 2 + y ** 2 / b ** 2 + z ** 2 / c ** 2)

        print(ellipse_c)

        # Check if the point is on the ellipsoid (with a tolerance due to possible floating point errors)
        assert np.isclose(ellipse_c, 1, atol=1e-6)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
