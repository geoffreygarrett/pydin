#include <iostream>

#include <pybind11/pybind11.h>

#include <pydin/bind_astro.hpp>
#include <pydin/bind_logging.hpp>
//#include <pydin/bind_gravitation.hpp>
#include <pydin/bind_mcts.hpp>
//#include <pydin/bind_omp.hpp>

namespace py = pybind11;
using namespace pybind11::literals;

#define ARG_ANGLE py::arg("angle")
#define ARG_GM py::arg("mu")


PYBIND11_MODULE(core, m) {
//    auto m_omp = m.def_submodule("omp");
//    bind_omp(m_omp, ""); https://stackoverflow.com/questions/69745880/how-can-i-multithread-this-code-snippet-in-c-with-eigen

#ifdef PYDIN_VERSION
    m.attr("__version__") = PYDIN_VERSION;
//    m.attr("_pydin_version_major") = PYDIN_VERSION_MAJOR;
//    m.attr("_pydin_version_minor") = PYDIN_VERSION_MINOR;
//    m.attr("_pydin_version_patch") = PYDIN_VERSION_PATCH;
#else
    m.attr("__version__") = "dev";
//    m.attr("_pydin_version_major") = 0;
//    m.attr("_pydin_version_minor") = 0;
//    m.attr("_pydin_version_patch") = 0;
#endif


    auto m_logging = m.def_submodule("logging");
    bind_logging(m_logging, "");

//    auto m_mip = m.def_submodule("mip");
//    auto m_mcts = m.def_submodule("mcts");

    using StateVariant = std::variant<
            state::mixed_integer_program<int, double>,
            Eigen::VectorX<int>>;

    py::class_<StateVariant>(m, "StateVariant", py::module_local())
        .def(py::init<>())
        .def(py::init<state::mixed_integer_program<int, double>>())
        .def(py::init<Eigen::VectorX<int>>())
        .def("is_mip", [](const StateVariant &s) {
        return std::holds_alternative<state::mixed_integer_program<int, double>>(s);
        }, "Returns true if the variant is a mixed integer program")
        .def("is_eigen", [](const StateVariant &s) {
        return std::holds_alternative<Eigen::VectorX<int>>(s);
        }, "Returns true if the variant is an eigen vector")
        .def("get_mip", [](const StateVariant &s) -> state::mixed_integer_program<int, double> {
        if (auto val = std::get_if<state::mixed_integer_program<int, double>>(&s)) {
        return *val;
        }
        throw std::runtime_error("StateVariant does not hold a mixed integer program");
        }, "Returns the mixed integer program")
        .def("get_eigen", [](const StateVariant &s) -> Eigen::VectorX<int> {
        if (auto val = std::get_if<Eigen::VectorX<int>>(&s)) {
        return *val;
        }
        throw std::runtime_error("StateVariant does not hold an Eigen vector");
        }, "Returns the Eigen vector")
        .def("__repr__", [](const StateVariant &s) {
        if (std::holds_alternative<state::mixed_integer_program<int, double>>(s)) {
        return "<StateVariant: mixed integer program>";
        } else if (std::holds_alternative<Eigen::VectorX<int>>(s)) {
        return "<StateVariant: Eigen vector>";
        } else {
        return "<StateVariant: unknown>";
        }
    });

    bind_mcts<
//            state::mixed_integer_program<int, double>,// tuple<vector<int>, vector<double>>
            StateVariant,// tuple<vector<int>, vector<double>>
            action::single_integer<int>,              // int
            reward::single_scalar<float>,             // float
            double>(m, "");

////
//    bind_mcts<
//            Eigen::VectorX<int>,         // vector<int> (shouldn't be eigen, but I have examples using this rn)
//            action::single_integer<int>, // int
//            reward::single_scalar<float>,// float
//            double>(m, "");


    bind_astrodynamics<double>(m, "");

//    auto m_gravitation = m.def_submodule("gravitation");

//    bind_gravitation<double>(m_gravitation, "");
}
