#ifndef BIND_GRAVITATION_HPP
#define BIND_GRAVITATION_HPP

#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include <odin/models/gravitational/ellipsoidal.hpp>
#ifdef USE_ESA_POLYHEDRAL_GRAVITY
#    include <odin/models/gravitational/polyhedral.hpp>
#endif

using namespace pybind11::literals;
namespace py = pybind11;
using namespace odin;


// TODO: Find some exception, or guard against this.. it causes the ellipsoid calculation to drift into oblivion?
//def test_potential_series():
//    ellipsoid = TriAxialEllipsoid(1.0, 1.0, 1.0, 1.0)
//    positions = np.array([[0.0, 0.0, 0.0],
//                          [1.0, 0.0, 0.0],
//                          [0.0, 1.0, 0.0],
//                          [0.0, 1.0, 0.0],
//                          [0.0, 1.0, 0.0],
//                          [0.0, 1.0, 0.0],
//                          [0.0, 1.0, 0.0],
//                          [0.0, 1.0, 0.0],
//                          [0.0, 1.0, 0.0],
//                          [0.0, 1.0, 0.0],
//
//                          ])
//    pdlog.info(f"Positions: {positions}")
//    potentials = ellipsoid.acceleration_series(positions)
//    pdlog.info(f"Potentials: {potentials}")
//    # assert potentials.shape[0] == positions.shape[0]

template<typename Scalar, typename T>
void add_gravitation_base_methods(py::class_<T> &cls) {
    using grid_2d_scalar_type = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using grid_3d_scalar_type = Eigen::Tensor<Scalar, 3>;

    cls.def("potential", &T::potential, "position"_a)
            .def("acceleration", &T::acceleration, "position"_a)
            .def("potential_series", &T::potential_series, "positions"_a)
            .def("acceleration_series", &T::acceleration_series, "positions"_a)
            .def(
                    "potential_grid",
                    [](T &instance, const std::array<grid_2d_scalar_type, 3> &positions)
                            -> grid_2d_scalar_type { return instance.potential_grid(positions); },
                    "grid_positions"_a)
            .def(
                    "potential_grid",
                    [](T &instance, const std::array<grid_3d_scalar_type, 3> &positions)
                            -> grid_3d_scalar_type { return instance.potential_grid(positions); },
                    "grid_positions"_a)
            .def("acceleration_grid", &T::acceleration_grid, "grid_positions"_a)
            .def("__copy__", &T::thread_local_copy);
}

//    using nodes_type            = typename polyhedral_type::nodes_type;
//    using faces_type            = typename polyhedral_type::faces_type;
//class Polyhedron:
//
//    def __init__(self, nodes, faces):
//        self.nodes = nodes
//        self.faces = faces
//
//    def check_face_orientation(self):
//        """Check if the face vertices are ordered clockwise."""
//        centroid = self.nodes.mean(axis=0)
//        for face in self.faces:
//            vec1, vec2 = self.nodes[face[1]] - self.nodes[face[0]], self.nodes[
//                face[2]] - self.nodes[face[0]]
//            normal, vec_centroid = np.cross(vec1, vec2), self.nodes[face[0]] - centroid
//            if np.dot(normal, vec_centroid) > 0:
//                return False
//        return True
//
//    def center_nodes_at_origin(self):
//        self.nodes -= self.nodes.mean(axis=0)
//
//    def reorient_faces(self, orientation="cw"):
//        """Reorient the faces so that the vertices are ordered based on orientation."""
//        assert orientation in ["cw", "ccw"], "Orientation should be 'cw' or 'ccw'."
//        is_cw = orientation == "cw"
//        for i, face in enumerate(self.faces):
//            if self.check_face_orientation() != is_cw:
//                self.faces[i] = face[::-1]
#include <array>
#include <vector>

using nodes_type = std::vector<std::array<double, 3>>;
using faces_type = std::vector<std::array<size_t, 3>>;


constexpr auto check_face_orientation = [](const nodes_type &nodes, const faces_type &faces) {
    std::array<double, 3> centroid = {0.0, 0.0, 0.0};

    for (const auto &node: nodes) {
        for (size_t i = 0; i < 3; ++i) {
            centroid[i] += node[i] / static_cast<double>(nodes.size());
        }
    }

    for (const auto &face: faces) {
        std::array<double, 3> vec1, vec2, normal, vec_centroid;

        for (size_t i = 0; i < 3; ++i) {
            vec1[i]         = nodes[face[1]][i] - nodes[face[0]][i];
            vec2[i]         = nodes[face[2]][i] - nodes[face[0]][i];
            vec_centroid[i] = nodes[face[0]][i] - centroid[i];
        }

        normal[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
        normal[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
        normal[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];

        double dot_product = normal[0] * vec_centroid[0] + normal[1] * vec_centroid[1]
                           + normal[2] * vec_centroid[2];
        if (dot_product > 0) { return false; }
    }

    return true;
};


template<typename Scalar>
void bind_gravitation(py::module &m, const std::string &suffix = "") {

    using scalar_type = Scalar;
    using vector_type = Eigen::Matrix<Scalar, 3, 1>;
    using matrix_type = Eigen::Matrix<Scalar, 3, 3>;


    //    using hollow_spherical_type = HollowSpherical<Scalar>;
    //    py::class_<hollow_spherical_type>(m, ("HollowSpherical" + suffix).c_str())
    //            .def(py::init<Scalar, Scalar>(), "radius"_a, "mu"_a)
    //            .def("potential", &hollow_spherical_type::potential, "position"_a)
    //            .def("acceleration", &hollow_spherical_type::acceleration, "position"_a)
    //            .def("potential_series", &hollow_spherical_type::potential_series, "positions"_a)
    //            .def("acceleration_series", &hollow_spherical_type::acceleration_series, "positions"_a)
    //            .def("potential_grid", &hollow_spherical_type::potential_grid, "grid_positions"_a)
    //            .def("acceleration_grid",
    //                 &hollow_spherical_type::acceleration_grid,
    //                 "grid_positions"_a)
    //            .def("radius", &hollow_spherical_type::radius)
    //            .def("mu", &hollow_spherical_type::mu);

#ifdef USE_GSL_ELLIPSOIDAL_GRAVITY
    py::class_<TriAxialEllipsoid<Scalar>> cls(m, "TriAxialEllipsoid");
    cls.def(py::init<Scalar, Scalar, Scalar, Scalar>(), "a"_a, "b"_a, "c"_a, "mu"_a);
    add_gravitation_base_methods<Scalar>(cls);
#endif

    //            .def(py::init<const typename polyhedral_type::polyhedron_type &, const Scalar &>(),
    //                 "polyhedron"_a,
    //                 "density"_a)
    //            .def(py::init<typename polyhedral_type::polyhedron_type &&, const Scalar &>(),
    //                 "polyhedron"_a,
    //                 "density"_a)
#ifdef USE_ESA_POLYHEDRAL_GRAVITY


    using polyhedron_shape_type = Polyhedron;
    using polyhedral_type       = Polyhedral<Scalar>;
    using nodes_type            = typename polyhedral_type::nodes_type;
    using faces_type            = typename polyhedral_type::faces_type;
    py::class_<polyhedron_shape_type, std::shared_ptr<polyhedron_shape_type>>(m, "PolyhedronShape")
            .def(py::init([](nodes_type &nodes, faces_type &faces) {
                     return std::make_shared<polyhedron_shape_type>(nodes, faces);
                 }),
                 "nodes"_a,
                 "faces"_a)
            //                .def(py::init<const typename polyhedron_shape_type &, const Scalar &>(),
            //                 "polyhedron"_a,
            //                 "density"_a)
            .def("are_normals_outward_pointing",
                 &polyhedralGravity::MeshChecking::checkNormalsOutwardPointing,
                 "Checks if the vertices are in such an order that the unit normals of each plane "
                 "point outwards the polyhedron. "
                 "Returns true if all the unit normals are pointing outwards.")
            .def("are_triangles_not_degenerated",
                 &polyhedralGravity::MeshChecking::checkTrianglesNotDegenerated,
                 "Checks if no triangle is degenerated by checking the surface area being greater "
                 "than zero. "
                 "Returns true if triangles are fine and none of them is degenerate.")
            .def("reorient_faces",
                 [&](
                         // nodes_type &nodes,
                         // faces_type &faces,
                         polyhedron_shape_type &self,
                         const std::string     &orientation = "cw") -> polyhedron_shape_type {
                     bool is_cw = (orientation == "cw");

                     auto faces = self.getFaces();// This will make a copy of the faces.
                     for (auto &face: faces) {
                         if (check_face_orientation(self.getVertices(), {face}) != is_cw) {
                             std::reverse(face.begin(), face.end());
                         }
                     }
                     // Then create a new polyhedron with the updated faces and return it
                     return {self.getVertices(), faces};
                 })
            .def("recenter_nodes_at_origin",
                 [](polyhedron_shape_type &self) {
                     // Create a copy of nodes since the original is const
                     nodes_type nodes = self.getVertices();

                     std::array<double, 3> centroid = {0.0, 0.0, 0.0};
                     for (const auto &node: nodes) {
                         for (size_t i = 0; i < 3; ++i) {
                             centroid[i] += node[i] / static_cast<double>(nodes.size());
                         }
                     }

                     for (auto &node: nodes) {
                         for (size_t i = 0; i < 3; ++i) { node[i] -= centroid[i]; }
                     }

                     // If you can modify vertices of self
                     // self.setVertices(nodes);

                     // Or return new object with modified nodes
                     // return polyhedron_shape_type(nodes, self.getFaces());
                 })
            .def("get_nodes",
                 [](polyhedron_shape_type &self) {
                     auto            eigen_nodes = self.getVertices();
                     Eigen::MatrixXd nodes(eigen_nodes.size(), 3);
                     for (size_t i = 0; i < eigen_nodes.size(); ++i) {
                         for (size_t j = 0; j < 3; ++j) { nodes(i, j) = eigen_nodes[i][j]; }
                     }
                     return nodes;
                 })
            .def("get_faces",
                 [](polyhedron_shape_type &self) {
                     auto                        eigen_faces = self.getFaces();
                     Eigen::MatrixX<std::size_t> faces(eigen_faces.size(), 3);
                     for (size_t i = 0; i < eigen_faces.size(); ++i) {
                         for (size_t j = 0; j < 3; ++j) { faces(i, j) = eigen_faces[i][j]; }
                     }
                     return faces;
                 })
            .def("count_faces", &polyhedron_shape_type::countFaces)
            .def("count_nodes", &polyhedron_shape_type::countVertices)
            .def("get_vertex", &polyhedron_shape_type::getVertex);


    py::class_<polyhedral_type> cls_polyhedral(m, ("Polyhedral" + suffix).c_str());


    using polyhedral_type = Polyhedral<Scalar>;

    cls_polyhedral
            .def(py::init<std::vector<std::string> &, const Scalar &>(),
                 "file_paths"_a,
                 "density"_a)
            .def(py::init<nodes_type, faces_type, const Scalar>(),
                 "nodes"_a,
                 "faces"_a,
                 "density"_a)

            .def(py::init<std::string &, const Scalar &>(), "file_path"_a, "density"_a)
            .def(py::init([](py::object path_obj, const Scalar &density) {
                     std::string file_path
                             = py::str(py::module_::import("os").attr("fspath")(path_obj));
                     return new polyhedral_type(file_path, density);
                 }),
                 "file_path"_a.noconvert(),
                 "density"_a)
            .def(py::init([](py::list path_list, const Scalar &density) {
                     std::vector<std::string> file_paths;
                     for (py::handle path_obj: path_list) {
                         file_paths.emplace_back(
                                 py::str(py::module_::import("os").attr("fspath")(path_obj)));
                     }
                     return new polyhedral_type(std::move(file_paths), density);
                 }),
                 "file_paths"_a.noconvert(),
                 "density"_a);
    add_gravitation_base_methods<Scalar>(cls_polyhedral);

//            .def("tensor", &polyhedral_type::tensor_impl, "position"_a)
//            .def("evaluate", &polyhedral_type::evaluate_imp, "position"_a);
#endif

    //    py::class_ < HollowSphere < Scalar >> (m, "HollowSphere")
    //            .def(py::init <
    //                 Scalar,
    //                 Scalar,
    //                 Eigen::Vector3 < Scalar >
    //                 > (),
    //                 "r"_a,
    //                 "mu"_a,
    //                 "center"_a
    //            )
    //            .def("potential", &HollowSphere<Scalar>::potential,
    //                 "position"_a
    //            )
    //            .def("acceleration", &HollowSphere<Scalar>::acceleration,
    //                 "position"_a
    //            )
    //            .def("calculate_potentials", &HollowSphere<Scalar>::calculate_potentials,
    //                 "x_positions"_a,
    //                 "y_positions"_a,
    //                 "z_positions"_a
    //            )
    //            .def("calculate_accelerations", &HollowSphere<Scalar>::calculate_accelerations,
    //                 "x_positions"_a,
    //                 "y_positions"_a,
    //                 "z_positions"_a
    //            );
}

#endif//BIND_GRAVITATION_HPP