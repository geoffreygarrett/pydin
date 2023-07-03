#ifndef BIND_OMP_HPP
#define BIND_OMP_HPP

#include <odin/logging.hpp>
#include <omp.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>


void bind_omp(py::module &m, const std::string &suffix = "") {
    using namespace pybind11::literals;
    namespace py = pybind11;

    omp_set_max_active_levels(2);

    m.def(
            "get_max_active_levels", []() {
                return omp_get_max_active_levels();
            },
            "Get the maximum number of nested parallel regions");

    m.def(
            "set_max_active_levels", [](int max_levels) {
                omp_set_max_active_levels(max_levels);
            },
            "Set the maximum number of nested parallel regions", "max_levels"_a);
}

#endif// BIND_OMP_HPP