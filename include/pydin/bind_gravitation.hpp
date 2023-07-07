#ifndef BIND_GRAVITATION_HPP
#define BIND_GRAVITATION_HPP

#include <odin/models/gravitational/ellipsoidal.hpp>

#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

using namespace pybind11::literals;
namespace py = pybind11;


template<typename Float>
void bind_gravitation(py::module &m, const std::string &suffix = "") {
    py::class_ < TriAxialEllipsoid < Float >> (m, "TriAxialEllipsoid")
            .def(py::init<
                         Float,
                         Float,
                         Float,
                         Float
                 >(),
                 "a"_a,
                 "b"_a,
                 "c"_a,
                 "mu"_a
            )
            .def("potential", &TriAxialEllipsoid<Float>::potential,
                 "position"_a
            )
            .def("acceleration", &TriAxialEllipsoid<Float>::acceleration,
                 "position"_a
            )
            .def("calculate_potentials", &TriAxialEllipsoid<Float>::calculate_potentials,
                 "x_positions"_a,
                 "y_positions"_a,
                 "z_positions"_a,
                 py::return_value_policy::move

            )
            .def("calculate_accelerations", &TriAxialEllipsoid<Float>::calculate_accelerations,
                 "x_positions"_a,
                 "y_positions"_a,
                 "z_positions"_a,
                    // reference
                 py::return_value_policy::move
            )
            .def("__copy__", [](const TriAxialEllipsoid<Float> &self) {
                return TriAxialEllipsoid<Float>(self);
            });

//    py::class_ < HollowSphere < Float >> (m, "HollowSphere")
//            .def(py::init <
//                 Float,
//                 Float,
//                 Eigen::Vector3 < Float >
//                 > (),
//                 "r"_a,
//                 "mu"_a,
//                 "center"_a
//            )
//            .def("potential", &HollowSphere<Float>::potential,
//                 "position"_a
//            )
//            .def("acceleration", &HollowSphere<Float>::acceleration,
//                 "position"_a
//            )
//            .def("calculate_potentials", &HollowSphere<Float>::calculate_potentials,
//                 "x_positions"_a,
//                 "y_positions"_a,
//                 "z_positions"_a
//            )
//            .def("calculate_accelerations", &HollowSphere<Float>::calculate_accelerations,
//                 "x_positions"_a,
//                 "y_positions"_a,
//                 "z_positions"_a
//            );
}

#endif //BIND_GRAVITATION_HPP