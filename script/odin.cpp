#include <pybind11/pybind11.h>

#include <iostream>

#include <pydin/bind_astro.hpp>
//#include <pydin/bind_autodiff.hpp>
#include <pydin/bind_gravitation.hpp>
#include <pydin/bind_jit.hpp>
#include <pydin/bind_linalg.hpp>
#include <pydin/bind_logging.hpp>
#include <pydin/bind_mcts.hpp>
#include <pydin/bind_omp.hpp>
#include <pydin/bind_tbb.hpp>
#include <pydin/bind_tree.hpp>

#ifdef ODIN_USE_GSL


#endif

namespace py = pybind11;
using namespace pybind11::literals;

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


}
