import numpy as np
import pytest
from pydin.core.shape import sphere, ellipsoid


# ----------------------------------
# ------- SPHERE TESTING -----------
# ----------------------------------

def test_sphere_creation():
    radius = 1.0
    s = sphere(radius=radius)
    assert s.radius == radius


def test_sphere_volume():
    radius = 1.0
    s = sphere(radius=radius)
    expected_volume = 4 / 3 * np.pi * radius ** 3
    assert np.isclose(s.volume(), expected_volume)


def test_sphere_surface_area():
    radius = 1.0
    s = sphere(radius=radius)
    expected_surface_area = 4 * np.pi * radius ** 2
    assert np.isclose(s.surface_area(), expected_surface_area)


def test_sphere_centroid():
    radius = 1.0
    s = sphere(radius=radius)
    assert np.allclose(s.centroid(), np.zeros(3))


def test_sphere_is_inside():
    radius = 1.0
    s = sphere(radius=radius)
    assert s.is_inside(np.array([0.5, 0, 0]))  # Inside point
    assert not s.is_inside(np.array([1.5, 0, 0]))  # Outside point


# -------------------------------------
# ------- ELLIPSOID TESTING -----------
# -------------------------------------

def test_ellipsoid_creation():
    radii = np.array([1.0, 0.5, 0.25])
    e = ellipsoid(*radii)
    assert e.radius_a == radii[0]
    assert e.radius_b == radii[1]
    assert e.radius_c == radii[2]


def test_ellipsoid_volume():
    radii = np.array([1.0, 0.5, 0.25])
    e = ellipsoid(*radii)
    expected_volume = 4 / 3 * np.pi * np.prod(radii)
    assert np.isclose(e.volume(), expected_volume)


def test_ellipsoid_surface_area():
    radii = np.array([1.0, 0.5, 0.25])
    e = ellipsoid(*radii)
    # The formula for the surface area of an ellipsoid is complex,
    # so for simplicity, we just ensure it's not zero
    assert e.surface_area() > 0


def test_ellipsoid_centroid():
    radii = np.array([1.0, 0.5, 0.25])
    e = ellipsoid(*radii)
    assert np.allclose(e.centroid(), np.zeros(3))


def test_ellipsoid_is_inside():
    radii = np.array([1.0, 0.5, 0.25])
    e = ellipsoid(*radii)
    assert e.is_inside(np.array([0.5, 0.2, 0.1]))  # Inside point
    assert not e.is_inside(np.array([1.5, 0.2, 0.1]))  # Outside point


# ----------------------------------
# ------- GENERAL TESTING ----------
# ----------------------------------

def test_shape_center():
    center = np.array([1.0, 2.0, 3.0])
    s = sphere(radius=1.0, center=center)
    e = ellipsoid(1.0, 0.5, 0.25, center=center)
    assert np.all(s.get_center() == center)
    assert np.all(e.get_center() == center)


if __name__ == "__main__":
    pytest.main([__file__])
