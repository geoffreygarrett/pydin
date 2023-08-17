from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import trimesh
from pydin.core.gravitation import PolyhedronShape, Polyhedral
from pydin.core.shape import sphere, ellipsoid


# create fixtures for shapes

@pytest.fixture
def sphere_instance():
    return sphere(radius=1.0)


@pytest.fixture
def ellipsoid_instance():
    return ellipsoid(*(3 * np.array([1.0, 0.2, 2.0])))


@pytest.fixture
def polyhedron_shape_instance():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return PolyhedronShape(nodes, faces)


# create tests for each functionality

def test_sphere_show(sphere_instance):
    # Mocking `plot_ellipsoid_mayavi`
    with patch("pydin.core.gravitation.plot_ellipsoid_mayavi", return_value=True) as mock_func:
        sphere_instance.show()
        mock_func.assert_called_once_with(sphere_instance.radius, sphere_instance.radius, sphere_instance.radius)


def test_ellipsoid_show(ellipsoid_instance):
    # Mocking `plot_ellipsoid_mayavi`
    with patch("pydin.core.gravitation.plot_ellipsoid_mayavi", return_value=True) as mock_func:
        ellipsoid_instance.show()
        mock_func.assert_called_once_with(ellipsoid_instance.radius_a, ellipsoid_instance.radius_b,
                                          ellipsoid_instance.radius_c)


def test_polyhedral_model(polyhedron_shape_instance):
    # Mocking Polyhedral
    mock_polyhedral = MagicMock()
    with patch("pydin.core.gravitation.Polyhedral", return_value=mock_polyhedral) as mock_class:
        model_params = {'nodes': polyhedron_shape_instance.nodes, 'faces': polyhedron_shape_instance.faces,
                        'density': 10000.0}
        model = Polyhedral(**model_params)
        mock_class.assert_called_once_with(**model_params)


# Test ellipsoid meshing using trimesh library

def test_ellipsoid_mesh():
    radii = np.array([3.0, 2.0, 1.0])
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    ellipsoid_mesh = sphere.apply_transform(np.diag(np.append(radii, 1)))
    assert ellipsoid_mesh.vertices.shape[1] == 3, "Vertices of the mesh are not 3D"
    assert ellipsoid_mesh.faces.shape[1] == 3, "Faces of the mesh are not triangular"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
