#ifndef PYDIN_SERIALIZATION_HPP
#define PYDIN_SERIALIZATION_HPP

#include <pybind11/pybind11.h>

#include <string>


namespace py = pybind11;

namespace cereal {
    template<class Archive>
    void save(Archive &ar, const py::object &obj) {
        // this might be too crude lol
        auto pickled = py::module::import("pickle").attr("dumps")(obj).cast<std::string>();
        ar(pickled);
    }

    template<class Archive>
    void load(Archive &ar, py::object &obj) {
        std::string pickled;
        ar(pickled);
        obj = py::module::import("pickle").attr("loads")(pickled);
    }
}// namespace cereal

#endif//PYDIN_SERIALIZATION_HPP
