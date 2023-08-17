#ifndef BIND_LINALG_HPP
#define BIND_LINALG_HPP

#include <pybind11/chrono.h>// for chrono
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>// for std::variant

#include <memory>
#include <thread>

#include <odin/io.hpp>
#include <odin/vectorize.hpp>
#include <pydin/common_macros.hpp>
#include <pydin/serialization.hpp>

namespace py = pybind11;
using namespace py::literals;


enum Framework { STDLIB, EIGEN };

Eigen::VectorXd linspace_eigen(double start, double stop, std::size_t num) {
    Eigen::VectorXd linspaced(num);

    if (num > 1) {
        double step = (stop - start) / static_cast<double>(num - 1);
        linspaced   = Eigen::VectorXd::LinSpaced(num, start, stop);
    } else {
        linspaced[0] = start;
    }

    return linspaced;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> meshgrid_eigen(const Eigen::VectorXd &x,
                                                           const Eigen::VectorXd &y) {
    Eigen::MatrixXd X = x.replicate(1, y.size());
    Eigen::MatrixXd Y = y.transpose().replicate(x.size(), 1);
    return std::make_pair(X, Y);
}

template<typename Float = double>
void bind_linalg(py::module &m, const std::string &suffix = "") {

    bind_io<Eigen::Matrix<Float, Eigen::Dynamic, 1>>(m, suffix, false, false, false, true);
    bind_io<std::vector<Float>>(m, suffix, false, false, false, true);

    m.def("linspace_eigen", &linspace_eigen, "Generate a linearly spaced array using Eigen.",
          "start"_a, "stop"_a, "num"_a);
    m.def("meshgrid_eigen", &meshgrid_eigen, "Generate a meshgrid using Eigen.");

    auto m_eigen = m.def_submodule("eigen", "Eigen linear algebra functions.");
    m_eigen.def("linspace", &linspace_eigen, "Generate a linearly spaced array.");
    m_eigen.def("meshgrid", &meshgrid_eigen, "Generate a meshgrid.");


    // Define the enumeration for namespace choices
    py::enum_<Framework>(m, "Framework")
            .value("STDLIB", Framework::STDLIB)
            .value("EIGEN", Framework::EIGEN)
            .export_values();

    m.def(
            "linspace",
            [](double      start,
               double      stop,
               std::size_t num,
               bool        endpoint = true,
               bool        parallel = false) -> Eigen::VectorXd {
                return odin::eigen::linspace(start, stop, num, endpoint, parallel);
            },
            "start"_a,
            "stop"_a,
            "num"_a,
            "endpoint"_a = true,
            "parallel"_a = false);

    m.def(
            "logspace",
            [](double      start,
               double      stop,
               std::size_t num,
               bool        endpoint = true,
               bool        parallel = false) -> Eigen::VectorXd {
                return odin::eigen::logspace(start, stop, num, endpoint, parallel);
            },
            "start"_a,
            "stop"_a,
            "num"_a,
            "endpoint"_a = true,
            "parallel"_a = false);

    m.def(
            "geomspace",
            [](double      start,
               double      stop,
               std::size_t num,
               bool        endpoint = true,
               bool        parallel = false) -> Eigen::VectorXd {
                return odin::eigen::geomspace(start, stop, num, endpoint, parallel);
            },
            "start"_a,
            "stop"_a,
            "num"_a,
            "endpoint"_a = true,
            "parallel"_a = false);
    // Define enumeration in the module
    py::enum_<odin::Indexing>(m, "Indexing")
            .value("xy", odin::Indexing::xy)
            .value("ij", odin::Indexing::ij)
            .export_values();

    // Define meshgrid function overloads
    m.def(
            "meshgrid",
            [](const Eigen::VectorXd &x,
               const Eigen::VectorXd &y,
               const odin::Indexing   indexing,
               bool                   parallel) { return meshgrid(indexing, parallel, x, y); },
            "x"_a,
            "y"_a,
            "indexing"_a = odin::Indexing::xy,
            "parallel"_a = true);

    m.def(
            "meshgrid",
            [](const Eigen::VectorXd &x,
               const Eigen::VectorXd &y,
               const Eigen::VectorXd &z,
               const odin::Indexing   indexing,
               bool                   parallel) { return meshgrid(indexing, parallel, x, y, z); },
            "x"_a,
            "y"_a,
            "z"_a,
            "indexing"_a = odin::Indexing::xy,
            "parallel"_a = true);

    m.def(
            "meshgrid",
            [](const Eigen::VectorXd x,
               const Eigen::VectorXd y,
               const Eigen::VectorXd z,
               const Eigen::VectorXd w,
               const odin::Indexing  indexing,
               bool parallel) { return meshgrid(indexing, parallel, x, y, z, w); },
            "x"_a,
            "y"_a,
            "z"_a,
            "w"_a,
            "indexing"_a = odin::Indexing::xy,
            "parallel"_a = true);
}

#endif//BIND_LINALG_HPP
