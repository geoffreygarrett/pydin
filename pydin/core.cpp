#include <pybind11/pybind11.h>

#include <iostream>

#include <pydin/bind_astro.hpp>
//#include <pydin/bind_autodiff.hpp>
#include <pydin/bind_gravitation.hpp>
//#include <pydin/bind_jit.hpp>
#include <pydin/bind_linalg.hpp>
#include <pydin/bind_logging.hpp>
#include <pydin/bind_mcts.hpp>
#include <pydin/bind_omp.hpp>
#include <pydin/bind_shape.h>
#include <pydin/bind_tbb.hpp>
#include <pydin/bind_tree.hpp>

#ifdef ODIN_USE_GSL


#endif

namespace py = pybind11;
using namespace pybind11::literals;
using namespace pydin;

PYBIND11_MODULE(core, m) {


    auto m_omp = m.def_submodule("omp");
    bind_omp(m_omp, "");
    // https://stackoverflow.com/questions/69745880/how-can-i-multithread-this-code-snippet-in-c-with-eigen

#ifdef PYDIN_VERSION
    m.attr("__version__") = PYDIN_VERSION;
    // m.attr("_pydin_version_major") = PYDIN_VERSION_MAJOR;
    // m.attr("_pydin_version_minor") = PYDIN_VERSION_MINOR;
    // m.attr("_pydin_version_patch") = PYDIN_VERSION_PATCH;
#else
    m.attr("__version__") = "dev";
// m.attr("_pydin_version_major") = 0;
// m.attr("_pydin_version_minor") = 0;
// m.attr("_pydin_version_patch") = 0;
#endif

    auto m_tbb = m.def_submodule("tbb");
    bind_tbb(m_tbb);

    // with pydin.tbb.TBBControl(4) as tbb:
    // print(tbb.get_max_threads())  # Should print '4'
    // print(tbb.get_stack_size())  # Should print the default value
//    bind_odinscript(m);

    auto m_logging = m.def_submodule("logging");
    bind_logging(m_logging, "");


    using StateVariant
            = std::variant<state::mixed_integer_program<int, double>, Eigen::VectorX<int>>;
    py::class_<StateVariant>(m, "StateVariant", py::module_local())
            .def(py::init<>())
            .def(py::init<state::mixed_integer_program<int, double>>())
            .def(py::init<Eigen::VectorX<int>>())
            .def(
                    "is_mip",
                    [](const StateVariant &s) {
                        return std::holds_alternative<state::mixed_integer_program<int, double>>(
                                s);
                    },
                    "Returns true if the variant is a mixed integer program")
            .def(
                    "is_eigen",
                    [](const StateVariant &s) {
                        return std::holds_alternative<Eigen::VectorX<int>>(s);
                    },
                    "Returns true if the variant is an eigen vector")
            .def(
                    "get_mip",
                    [](const StateVariant &s) -> state::mixed_integer_program<int, double> {
                        if (auto val
                            = std::get_if<state::mixed_integer_program<int, double>>(&s)) {
                            return *val;
                        }
                        throw std::runtime_error(
                                "StateVariant does not hold a mixed integer program");
                    },
                    "Returns the mixed integer program")
            .def(
                    "get_eigen",
                    [](const StateVariant &s) -> Eigen::VectorX<int> {
                        if (auto val = std::get_if<Eigen::VectorX<int>>(&s)) { return *val; }
                        throw std::runtime_error("StateVariant does not hold an Eigen vector");
                    },
                    "Returns the Eigen vector")
            .def("__repr__", [](const StateVariant &s) {
                if (std::holds_alternative<state::mixed_integer_program<int, double>>(s)) {
                    return "<StateVariant: mixed integer program>";
                } else if (std::holds_alternative<Eigen::VectorX<int>>(s)) {
                    return "<StateVariant: Eigen vector>";
                } else {
                    return "<StateVariant: unknown>";
                }
            });

    bind_mcts<StateVariant, action::single_integer<int>, reward::single_scalar<float>, double>(m,
                                                                                               "");

    // bind_mcts<Eigen::VectorX<int>, action::single_integer<int>, reward::single_scalar<float>,
    // double>(m, "");

    bind_astrodynamics<double>(m, "");

    auto m_gravitation  = m.def_submodule("gravitation");
    m_gravitation.doc() = R"pbdoc(
            Gravitation submodule
            ---------------------
            .. currentmodule:: gravitation
            .. autosummary::
               :toctree: _generate
               gravitation
        )pbdoc";

    auto m_tree = m.def_submodule("tree");
    bind_tree<>(m_tree, "");

    auto m_shape = m.def_submodule("shape");
    bind_shape<double, Eigen::Vector3d>(m_shape);

    auto m_linalg = m.def_submodule("linalg");
    bind_linalg<>(m_linalg, "");

    //    auto m_autodiff = m.def_submodule("autodiff");
    //    bind_autodiff<>(m_autodiff, "");

#ifdef ODIN_USE_GSL
    bind_gravitation<double>(m_gravitation, "");
#endif
}
