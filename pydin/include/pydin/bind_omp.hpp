#ifndef BIND_OMP_HPP
#define BIND_OMP_HPP

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <omp.h>
#include <string>

#include <odin/logging.hpp>


class OMPThreadManager {
public:
    OMPThreadManager(int num_threads)
        : original_num_threads(omp_get_num_threads()), requested_num_threads(num_threads) {}

    void enter() { omp_set_num_threads(requested_num_threads); }

    void exit(py::args) { omp_set_num_threads(original_num_threads); }

private:
    int original_num_threads;
    int requested_num_threads;
};

void bind_omp(py::module &m, const std::string &suffix = "") {
    using namespace pybind11::literals;
    namespace py = pybind11;

    // Default max active levels to the number of processors
    int num_procs = omp_get_num_procs();
    omp_set_max_active_levels(num_procs);

    m.def(
            "get_max_active_levels",
            []() { return omp_get_max_active_levels(); },
            "Get the maximum number of nested parallel regions");

    m.def(
            "set_max_active_levels",
            [](int max_levels) { omp_set_max_active_levels(max_levels); },
            "Set the maximum number of nested parallel regions",
            "max_levels"_a);

    m.def(
            "get_num_threads",
            []() { return omp_get_num_threads(); },
            "Get the current number of threads");

    m.def(
            "set_num_threads",
            [](int num_threads) { omp_set_num_threads(num_threads); },
            "Set the number of threads",
            "num_threads"_a);

    py::class_<OMPThreadManager>(m, "OMPThreadManager")
            .def(py::init<int>(), "num_threads"_a = omp_get_num_procs())
            .def("__enter__", &OMPThreadManager::enter)
            .def("__exit__", &OMPThreadManager::exit);
}


#endif// BIND_OMP_HPP