#ifndef ODIN_BIND_AUTODIFF_H
#define ODIN_BIND_AUTODIFF_H


#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/reverse/var.hpp>
namespace py = pybind11;

template<typename Scalar = double>
void bind_dual(py::module &m, const std::string &suffix = "") {
    py::class_<autodiff::dual>(m, "dual")
            .def(py::init<double>())
            .def(
                    "__add__",
                    [](const autodiff::dual &self, const autodiff::dual &other) {
                        return self + other;
                    },
                    py::is_operator())
            .def(
                    "__mul__",
                    [](const autodiff::dual &self, const autodiff::dual &other) {
                        return self * other;
                    },
                    py::is_operator())
            .def(
                    "__truediv__",
                    [](const autodiff::dual &self, const autodiff::dual &other) {
                        return self / other;
                    },
                    py::is_operator())
            // Bind other dual methods and operators...
            ;
}
//
//void bind_var(py::module &m, const std::string &suffix = "") {
//    py::class_<autodiff::var>(m, "Var")
//            .def(py::init<double>())
//            .def(
//                    "__add__",
//                    [](const autodiff::var &self, const autodiff::var &other) {
//                        return self + other;
//                    },
//                    py::is_operator())
//            .def(
//                    "__mul__",
//                    [](const autodiff::var &self, const autodiff::var &other) {
//                        return self * other;
//                    },
//                    py::is_operator())
//            .def(
//                    "__truediv__",
//                    [](const autodiff::var &self, const autodiff::var &other) {
//                        return self / other;
//                    },
//                    py::is_operator())
//            // Bind other var methods and operators...
//            ;
//}

//template<typename Scalar = double>
//void bind_autodiff_functions(py::module &m, const std::string &suffix = "") {
//    m.def("derivative",
//          &autodiff::derivative,
//          "Compute the derivative of a function at a given point");
//    m.def("derivatives",
//          &autodiff::derivatives,
//          "Compute all derivatives of a function at a given point");
//    // Bind other autodiff functions...
//}

template<typename Scalar = double>
void bind_autodiff(py::module &m, const std::string &suffix = "") {
    bind_dual<Scalar>(m);
    //    bind_var<Scalar>(m);
//    bind_autodiff_functions<Scalar>(m);
}


#endif//ODIN_BIND_AUTODIFF_H