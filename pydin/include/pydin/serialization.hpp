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


template<typename T>
void bind_io(py::module        &m,
             const std::string &suffix   = "",
             bool               portable = false,
             bool               binary   = false,
             bool               json     = false,
             bool               xml      = false) {
    if (json) {
        m.def("to_json", &odin::to_json<T>, "object"_a);
        m.def("from_json", &odin::from_json<T>, "json_str"_a);
        m.def("load_json", &odin::load_json<T>, "filename"_a);
        m.def("save_json", &odin::save_json<T>, "object"_a, "filename"_a);
    }
    if (xml) {
        m.def("load_xml", &odin::load_xml<T>, "filename"_a);
        m.def("save_xml", &odin::save_xml<T>, "object"_a, "filename"_a);
        m.def("to_xml", &odin::to_xml<T>, "object"_a);
        m.def("from_xml", &odin::from_xml<T>, "xml_str"_a);
    }
    if (binary) {
        m.def("save_binary",
              &odin::save_binary<T>,
              "object"_a,
              "filename"_a,
              "portable"_a = portable);
        m.def("load_binary", &odin::load_binary<T>, "filename"_a, "portable"_a = portable);
        m.def("to_binary", &odin::to_binary<T>, "object"_a, "portable"_a = portable);
        m.def("from_binary", &odin::from_binary<T>, "binary_str"_a, "portable"_a = portable);
    }
}

#endif//PYDIN_SERIALIZATION_HPP
