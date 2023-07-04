#include <iostream>

#include <pybind11/pybind11.h>

#include <pydin/bind_astro.hpp>
#include <pydin/bind_logging.hpp>
#include <pydin/bind_mcts.hpp>
//#include <pydin/bind_omp.hpp>

namespace py = pybind11;
using namespace pybind11::literals;

#define ARG_ANGLE py::arg("angle")
#define ARG_GM py::arg("mu")


PYBIND11_MODULE(core, m) {
//    auto m_omp = m.def_submodule("omp");
//    bind_omp(m_omp, "");

    auto m_logging = m.def_submodule("logging");
    bind_logging(m_logging, "");

    bind_running_stats<double>(m, "");

    auto m_mip = m.def_submodule("mip");

    bind_mcts<
            state::mixed_integer_program<int, double>,// tuple<vector<int>, vector<double>>
            action::single_integer<int>,              // int
            reward::single_scalar<float>,             // float
            double>(m_mip, "");

    bind_mcts<
            Eigen::VectorX<int>,         // vector<int> (shouldn't be eigen, but I have examples using this rn)
            action::single_integer<int>, // int
            reward::single_scalar<float>,// float
            double>(m, "");


    bind_astrodynamics<double>(m, "");
}
