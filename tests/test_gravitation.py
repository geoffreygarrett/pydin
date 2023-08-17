import os

import numpy as np
import pytest
from pydin.core.gravitation import TriAxialEllipsoid, Polyhedral

# Bazel sets this environment variable
runfiles_dir = os.environ.get('RUNFILES_DIR')
eros_50k_path = os.path.join(runfiles_dir, 'eros_50k_ply/file/eros_50k.ply')
eros_node_path = os.path.join(runfiles_dir, 'Eros.node')
eros_face_path = os.path.join(runfiles_dir, 'Eros.face')

# check if the file exists
if not os.path.isfile(eros_50k_path):
    raise FileNotFoundError(f"File {eros_50k_path} not found")

# import pydin.core.logging as pdlog
# pdlog.set_level(pdlog.DEBUG)

# Define parameters
a, b, c = 300.0, 200.0, 100.0
rho = 2.8 * 1000.0
G = 6.67408 * 1e-11
mu = 4.0 / 3.0 * np.pi * G * rho * a * b * c

# Define parameters for each model
parameters_dict = {
    TriAxialEllipsoid: dict(a=300.0, b=200.0, c=100.0, mu=4.0 / 3.0 * np.pi * G * rho * a * b * c),
    # ModelA: dict(param1=300.0, param2=200.0),  # replace with actual parameters for ModelA
    # ModelB: dict(param1=100.0, param2=50.0, param3=25.0)  # replace with actual parameters for ModelB
}


# Parametrize the fixture with the keys (models) in the parameters_dict
@pytest.fixture(params=parameters_dict.keys())
def gravity(request):
    model = request.param
    return model(**parameters_dict[model])


def test_symmetry_potential(gravity):
    pos1 = np.array([400.0, 400.0, 400.0])
    pos2 = -1 * pos1
    assert gravity.potential(pos1) == pytest.approx(gravity.potential(pos2), rel=1e-6)


def test_symmetry_acceleration(gravity):
    pos1 = np.array([400.0, 400.0, 400.0])
    pos2 = -1 * pos1
    acc1 = gravity.acceleration(pos1)
    acc2 = gravity.acceleration(pos2)
    assert np.allclose(acc1, -acc2, rtol=1e-6)


# def test_known_potential_value(gravity):
#     pos = np.array([1000.0, 0.0, 0.0])
#     known_potential = -0.006774347691451479  # replace with the known value
#     assert gravity.potential(pos) == pytest.approx(known_potential, rel=1e-6)


# def test_known_acceleration_value(gravity):
#     pos = np.array([1000.0, 0.0, 0.0])
#     known_acceleration = np.array([-0.00677, 0.0, 0.0])  # replace with the known value
#     assert np.allclose(gravity.acceleration(pos), known_acceleration, rtol=1e-6)


def test_potential_increase_with_distance(gravity):
    pos1 = np.array([400.0, 400.0, 400.0])
    pos2 = np.array([800.0, 800.0, 800.0])
    assert gravity.potential(pos1) < gravity.potential(pos2)


def test_acceleration_decrease_with_distance(gravity):
    pos1 = np.array([400.0, 400.0, 400.0])
    pos2 = np.array([800.0, 800.0, 800.0])
    assert np.linalg.norm(gravity.acceleration(pos1)) > np.linalg.norm(gravity.acceleration(pos2))


def test_potential_series(gravity):
    positions = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]]) * 1000.0
    potentials = gravity.potential_series(positions)
    assert potentials.shape[0] == positions.shape[0]


def test_acceleration_series(gravity):
    positions = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]]) * 1000.0
    accelerations = gravity.acceleration_series(positions)
    assert accelerations.shape == positions.shape


def test_potential_series_values(gravity):
    positions = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]]) * 1000.0
    potentials = gravity.potential_series(positions)
    expected = np.array([gravity.potential(p) for p in positions])
    assert np.allclose(potentials, expected, rtol=1e-6)


def test_acceleration_series_values(gravity):
    positions = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]]) * 1000.0
    accelerations = gravity.acceleration_series(positions)
    expected = np.array([gravity.acceleration(p) for p in positions])
    assert np.allclose(accelerations, expected, rtol=1e-6)


def test_potential_series_empty_input(gravity):
    # TODO: should this raise an error or return an empty array? probably the latter
    positions = np.array([])
    with pytest.raises(TypeError):
        gravity.potential_series(positions)


def test_potential_series_wrong_input_type(gravity):
    positions = "not a numpy array"
    with pytest.raises(TypeError):
        gravity.potential_series(positions)


def test_potential_grid_shape(gravity):
    positions = np.meshgrid(
        np.arange(0., 1000., 50),
        np.arange(0., 1000., 50),
        np.arange(0., 1000., 50),
        indexing='ij')
    potentials = gravity.potential_grid(positions)
    assert potentials.shape == positions[0].shape  # the shape of the potential grid should match the grid dimensions


def vectorize_3d_input(func):
    """Wrap a vector function to allow np.vectorize to handle 3D inputs"""

    def wrapped_func(x, y, z):
        return func(np.array([x, y, z]))

    return wrapped_func


def test_potential_grid_values(gravity):
    positions = np.meshgrid(
        np.arange(0., 1000., 50),
        np.arange(0., 1000., 50),
        np.arange(0., 1000., 50),
        indexing='ij')
    potentials = gravity.potential_grid(positions)

    vectorized_potential = np.vectorize(vectorize_3d_input(gravity.potential))
    expected = vectorized_potential(*positions)

    assert np.allclose(potentials, expected, rtol=1e-6)


def test_potential_grid_empty_input(gravity):
    positions = np.array([])
    with pytest.raises(TypeError):
        gravity.potential_grid(positions)


def test_potential_grid_wrong_input_type(gravity):
    positions = "not a numpy array"
    with pytest.raises(TypeError):
        gravity.potential_grid(positions)


# def test_load_polyhedral():
#     gravity = Polyhedral([eros_node_path, eros_face_path], 2675.)


# def test_load_polyhedral():
#     # Use pandas to load the data from the files
#     nodes = pd.read_csv(eros_node_path, header=None).values
#     faces = pd.read_csv(eros_face_path, header=None).values
#
#     # Convert pandas DataFrame to list of std::array
#     nodes_list = [list(map(float, node)) for node in nodes]
#     faces_list = [list(map(int, face)) for face in faces]
#
#     # Convert lists to numpy arrays
#     nodes_array = np.array(nodes_list)
#     faces_array = np.array(faces_list)
#
#     # Create the Polyhedral object
#     gravity = Polyhedral(nodes_array, faces_array, 2675.)
#
#     return gravity
#

def get_polyhedral_cube(length=500.):
    # Create nodes for a cube with side length of 500m.
    nodes = np.array([
        [0, 0, 0],
        [length, 0, 0],
        [500, length, 0],
        [0, length, 0],
        [0, 0, length],
        [500, 0, length],
        [length, length, length],
        [0, length, length]
    ])

    # Create faces for the cube
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Front face
        [2, 3, 7], [2, 7, 6],  # Back face
        [0, 3, 7], [0, 7, 4],  # Left face
        [1, 2, 6], [1, 6, 5]  # Right face
    ])

    return nodes, faces


def get_triangular_prism(length=500.):
    # Create nodes for a cube with side length of 500m.
    nodes = np.array([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length],
        [length, 0, length],
        [0, length, length]
    ])

    # Create faces for the cube
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [3, 4, 5], [3, 5, 2],  # Top face
        [0, 1, 4], [0, 4, 3],  # Front face
        [1, 2, 5], [1, 5, 4],  # Back face
        [0, 2, 5], [0, 5, 3],  # Left face
        [0, 1, 4], [0, 4, 3]  # Right face
    ])

    return nodes, faces


def test_polyhedral_cube():
    # nodes, faces = get_polyhedral_cube()
    nodes, faces = get_triangular_prism()
    polyhedral = Polyhedral(nodes, faces, 2675.)

    # contourplot

    # Add some assertions here depending on what you want to test.
    # For example, we could test the number of nodes and faces:
    # assert polyhedral.nodes.shape == (8, 3)
    # assert polyhedral.faces.shape == (12, 3)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
