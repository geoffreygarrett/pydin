from pydin.core.gravitation import TriAxialEllipsoid
import numpy as np
import pytest

# Define parameters
a, b, c = 300.0, 200.0, 100.0
rho = 2.8 * 1000.0
G = 6.67408 * 1e-11
mu = 4.0 / 3.0 * np.pi * G * rho * a * b * c


@pytest.fixture
def gravity():
    return TriAxialEllipsoid(a, b, c, mu)


def test_tri_axial_ellipsoid_no_error(gravity):
    p = gravity.potential(np.array([400.0, 400.0, 400.0]))
    assert p == pytest.approx(-0.006774347691451479, rel=1e-6)


def test_tri_axial_ellipsoid_not_fixed(gravity):
    # check two different positions and ensure that the potential is different
    p1 = gravity.potential(np.array([400.0, 400.0, 400.0]))
    p2 = gravity.potential(np.array([500.0, 500.0, 500.0]))
    assert p1 != p2


def test_tri_axial_ellipsoid_parallel_not_fixed(gravity):
    # check two different positions and ensure that the potential is the same
    x, y, z = np.meshgrid(np.linspace(-500, 500, 5), np.linspace(-500, 500, 5), np.linspace(-500, 500, 5))
    gravity1 = TriAxialEllipsoid(a, b, c, mu)
    gravity2 = TriAxialEllipsoid(a, b * 10, c, mu)
    potentials1 = gravity1.calculate_potentials(x, y, z)
    potentials2 = gravity2.calculate_potentials(x, y, z)
    # check that its not the same
    assert not np.allclose(potentials1, potentials2)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
