#ifndef BIND_GRAVITATION_HPP
#define BIND_GRAVITATION_HPP

#include <odin/models/gravitational/ellipsoidal.hpp>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

using namespace pybind11::literals;
namespace py = pybind11;


template<typename Float>
void bind_gravitation(py::module &m, const std::string &suffix = "") {
    py::class_<gsl_root_fsolver_type>(m, "gsl_root_fsolver_type");
    py::class_<gsl_root_fdfsolver_type>(m, "gsl_root_fdfsolver_type");
    py::class_<gsl_root_fsolver_brent>(m, "gsl_root_fsolver_brent");
    py::class_<gsl_root_fsolver_brent>(m, "gsl_root_fdfsolver_steffenson");
    py::class_ < TriAxialEllipsoid < Float >> (m, "TriAxialEllipsoid")
            .def(py::init<
                         Float,
                         Float,
                         Float,
                         Float,
                         gsl_root_fsolver_type,
                         gsl_root_fdfsolver_type
                 >(),
                 "a"_a,
                 "b"_a,
                 "c"_a,
                 "mu"_a,
                 "bracketing_solver_type"_a,
                 "derivative_solver_type"_a
            )
            .def("potential", &TriAxialEllipsoid<Float>::potential,
                 "position"_a
            )
//            .def("gradient", &TriAxialEllipsoid<Float>::gradient,
//                 "position"_a
//            )
//            .def("hessian", &TriAxialEllipsoid<Float>::hessian,
//                 "position"_a
//            )
            .def("acceleration", &TriAxialEllipsoid<Float>::acceleration,
                 "position"_a
            )
//            .def("jacobian", &TriAxialEllipsoid<Float>::jacobian,
//                 "position"_a
//            )



}

#endif //BIND_GRAVITATION_HPP