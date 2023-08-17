import numpy as np

from .. import attempt_import

gravitation = attempt_import('pydin.core.gravitation')


class Polyhedron:

    def __init__(self, nodes, faces):
        self.nodes = nodes
        self.faces = faces

    def check_face_orientation(self):
        """Check if the face vertices are ordered clockwise."""
        centroid = self.nodes.mean(axis=0)
        for face in self.faces:
            vec1, vec2 = self.nodes[face[1]] - self.nodes[face[0]], self.nodes[
                face[2]] - self.nodes[face[0]]
            normal, vec_centroid = np.cross(vec1, vec2), self.nodes[face[0]] - centroid
            if np.dot(normal, vec_centroid) > 0:
                return False
        return True

    def center_nodes_at_origin(self):
        self.nodes -= self.nodes.mean(axis=0)

    def reorient_faces(self, orientation="cw"):
        """Reorient the faces so that the vertices are ordered based on orientation."""
        assert orientation in ["cw", "ccw"], "Orientation should be 'cw' or 'ccw'."
        is_cw = orientation == "cw"
        for i, face in enumerate(self.faces):
            if self.check_face_orientation() != is_cw:
                self.faces[i] = face[::-1]


# if gravitation is None:
#     PolyhedronShape = Polyhedron
# else:
#     PolyhedronShape = gravitation.PolyhedronShape
PolyhedronShape = Polyhedron


# Cube Example
def get_polyhedral_cube(l=1.):
    # Define the nodes and faces for the cube
    _nodes = np.array([[0, 0, 0], [l, 0, 0], [l, l, 0], [0, l, 0], [0, 0, l], [l, 0, l], [l, l, l],
                       [0, l, l]])

    _faces = np.array([
        [0, 1, 2],
        [0, 2, 3],  # Bottom face
        [4, 5, 6],
        [4, 6, 7],  # Top face
        [0, 1, 5],
        [0, 5, 4],  # Front face
        [2, 3, 7],
        [2, 7, 6],  # Back face
        [0, 3, 7],
        [0, 7, 4],  # Left face
        [1, 2, 6],
        [1, 6, 5]  # Right face
    ])

    _cube = PolyhedronShape(_nodes, _faces)
    _cube.center_nodes_at_origin()
    _cube.reorient_faces(orientation="cw")

    return cube


# Pyramid Example
def get_pyramid(base_length=1., height=1.):
    # Define the nodes and faces for the pyramid
    nodes = np.array([
        [0, 0, 0],  # Node 0 (Base corner)
        [base_length, 0, 0],  # Node 1 (Base corner)
        [base_length, base_length, 0],  # Node 2 (Base corner)
        [0, base_length, 0],  # Node 3 (Base corner)
        [base_length / 2, base_length / 2, height]  # Node 4 (Apex)
    ])

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Base triangles
        [0, 1, 4],  # Side triangle
        [1, 2, 4],  # Side triangle
        [2, 3, 4],  # Side triangle
        [3, 0, 4],  # Side triangle
    ])

    pyramid = PolyhedronShape(nodes, faces)
    pyramid.center_nodes_at_origin()
    pyramid.reorient_faces(orientation="cw")

    return pyramid


# Prism Example
def get_right_angled_triangular_prism(l=1.):
    # Define the nodes and faces for the prism
    _nodes = np.array([
        [0, 0, 0],  # Node 0
        [l, 0, 0],  # Node 1
        [0, l, 0],  # Node 2
        [0, 0, l],  # Node 3
        [l, 0, l],  # Node 4
        [0, l, l],  # Node 5
    ])

    _faces = np.array([
        [0, 1, 2],  # Bottom triangle
        [3, 4, 5],  # Top triangle
        [0, 1, 4],
        [0, 4, 3],  # Front rectangle divided into two triangles
        [0, 2, 5],
        [0, 5, 3],  # Left rectangle divided into two triangles
        [1, 2, 5],
        [1, 5, 4],  # Right rectangle divided into two triangles
    ])

    _prism = PolyhedronShape(_nodes, _faces)
    _prism.center_nodes_at_origin()
    _prism.reorient_faces(orientation="cw")

    return _prism


def get_subdivided_isocahedron(a, b, c, subdivisions=3):
    t = (1.0 + np.sqrt(5.0)) / 2.0

    # Start with an icosahedron
    nodes = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ])

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ])

    # Subdivide each face into 4 faces
    for _ in range(subdivisions):
        faces2 = np.zeros((len(faces) * 4, 3), dtype=np.int64)
        nodes = np.concatenate([nodes, np.zeros((len(faces) * 3, 3))])

        for i, face in enumerate(faces):
            a, b, c = face
            ab = len(nodes) - len(faces) * 3 + i * 3
            ac = ab + 1
            bc = ab + 2

            nodes[ab] = (nodes[a] + nodes[b]) / 2
            nodes[ac] = (nodes[a] + nodes[c]) / 2
            nodes[bc] = (nodes[b] + nodes[c]) / 2

            faces2[i * 4 + 0] = [a, ab, ac]
            faces2[i * 4 + 1] = [b, bc, ab]
            faces2[i * 4 + 2] = [c, ac, bc]
            faces2[i * 4 + 3] = [ab, bc, ac]

        faces = faces2

    # Normalize nodes to lie on the surface of the ellipsoid
    for i in range(len(nodes)):
        x, y, z = nodes[i]
        nodes[i] = [a * x / np.sqrt(x ** 2 + y ** 2 + z ** 2),
                    b * y / np.sqrt(x ** 2 + y ** 2 + z ** 2),
                    c * z / np.sqrt(x ** 2 + y ** 2 + z ** 2)]

    # Create the polyhedron
    polyhedron = PolyhedronShape(nodes, faces)


if __name__ == "__main__":
    cube = get_polyhedral_cube()
    print(cube.nodes)
    print(cube.faces)

    prism = get_right_angled_triangular_prism()
    print(prism.nodes)
    print(prism.faces)
