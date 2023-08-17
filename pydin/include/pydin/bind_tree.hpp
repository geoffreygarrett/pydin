#ifndef BIND_TREE_HPP
#define BIND_TREE_HPP

#include <pybind11/chrono.h>// for chrono
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>// for std::variant

#include <memory>
#include <thread>

#include <odin/io.hpp>
#include <odin/tree/base.hpp>
#include <odin/tree/search.hpp>
#include <odin/tree/serialization.hpp>
#include <pydin/common_macros.hpp>
#include <pydin/serialization.hpp>

namespace py = pybind11;
using namespace py::literals;

struct NodeData {
    virtual ~NodeData() = default;
};

using data_variant_types = std::variant<int,
                                        float,
                                        std::string,
                                        py::object,
                                        Eigen::VectorXd,
                                        Eigen::MatrixXd,
                                        Eigen::VectorXi>;

using data_variant_vector = std::vector<data_variant_types>;
using data_variant        = std::variant<data_variant_types, data_variant_vector>;

template<typename Object, typename Func>
auto weak_ptr_cast_to_shared_or_none(Func func) {
    return [func](Object &self) -> py::object {
        auto weak_ptr_result = (self.*func)();
        return weak_ptr_result.lock() ? py::cast(*weak_ptr_result.lock()) : py::none();
    };
}
//
//template<typename T = data_variant>
//void bind_io(py::module        &m,
//             const std::string &suffix   = "",
//             bool               portable = false,
//             bool               binary   = false,
//             bool               json     = false,
//             bool               xml      = false) {
//    if (json) {
//        m.def("to_json", &to_json<T>, "object"_a);
//        m.def("from_json", &from_json<T>, "json_str"_a);
//        m.def("load_json", &load_json<T>, "filename"_a);
//        m.def("save_json", &save_json<T>, "object"_a, "filename"_a);
//    }
//    if (xml) {
//        m.def("load_xml", &load_xml<T>, "filename"_a);
//        m.def("save_xml", &save_xml<T>, "object"_a, "filename"_a);
//        m.def("to_xml", &to_xml<T>, "object"_a);
//        m.def("from_xml", &from_xml<T>, "xml_str"_a);
//    }
//    if (binary) {
//        m.def("save_binary", &save_binary<T>, "object"_a, "filename"_a, "portable"_a = portable);
//        m.def("load_binary", &load_binary<T>, "filename"_a, "portable"_a = portable);
//        m.def("to_binary", &to_binary<T>, "object"_a, "portable"_a = portable);
//        m.def("from_binary", &from_binary<T>, "binary_str"_a, "portable"_a = portable);
//    }
//}


template<typename T = data_variant, typename Float = double>
void bind_tree(py::module &m, const std::string &suffix = "") {

    // bind io functions for the data variants
    bind_io<T>(m,
               suffix,
               true, // binary default: portable
               true, // binary io methods
               true, // json io methods
               true);// xml io methods

    // bind io functions for the data variants
    bind_io<std::vector<T>>(m,
                            suffix,
                            true, // binary default: portable
                            true, // binary io methods
                            true, // json io methods
                            true);// xml io methods

    using node_type = odin::SafeNode<T>;
    py::class_<node_type, std::shared_ptr<node_type>>(m, ("Node" + suffix).c_str())
            .def(py::init<const T &>(), "data"_a)
            //            .def(py::init<const T &, const std::shared_ptr<node_type> &>(), "data"_a, "parent"_a)
            .def("add_child", &node_type::add_child, "child"_a)
            .def("get_data", &node_type::get_data)
            .def("get_children", &node_type::get_children)
            //            .def("get_parent_level", &node_type::get_parent_level)
            // they're stored as weak_ptr, so we need to do this (totally safe here)
            .def("get_parent", weak_ptr_cast_to_shared_or_none<node_type>(&node_type::get_parent))
            .def("get_level", &node_type::get_level)
            .def("has_parent", &node_type::has_parent)
            .def("set_parent", &node_type::set_parent, "parent"_a)
            .def("remove_child", &node_type::remove_child, "child"_a)
            .def("clone", &node_type::clone)

            // properties
            .def("get_node_count", &node_type::get_node_count)
            .def("get_max_depth", &node_type::get_max_depth)
            .DEF_IO_BINARY_METHODS(node_type, false)
            .DEF_IO_JSON_METHODS(node_type)
            .DEF_PY_BINARY_PICKLING(node_type)

            //            // add a function to check if the instance is a Python object
            //            .def("holds_python_object", [](const node_type &self) {
            //                const auto &data = self.get_data();
            //                return std::holds_alternative<py::object>(data);
            //            });
            // add a function to check if the instance is a Python object
            .def("holds_python_object", [](const node_type &self) {
                const auto &data = self.get_data();

                if (std::holds_alternative<data_variant_types>(data)) {
                    const auto &data_variant = std::get<data_variant_types>(data);
                    return std::holds_alternative<py::object>(data_variant);
                } else if (std::holds_alternative<data_variant_vector>(data)) {
                    const auto &data_vector = std::get<data_variant_vector>(data);
                    for (const auto &data_variant: data_vector) {
                        if (std::holds_alternative<py::object>(data_variant)) { return true; }
                    }
                }
                return false;
            });

    // bind the tree class
    using tree_type           = Tree<T>;
    using safe_node_type      = typename tree_type::safe_node_type;
    using raw_node_unique_ptr = typename tree_type::p_raw_node;
    using p_safe_node         = typename tree_type::p_safe_node;
    //
    py::class_<tree_type, std::shared_ptr<tree_type>>(m, ("Tree" + suffix).c_str())
            .def(py::init<const p_safe_node &>(), "root"_a)
            .def("get_root", &tree_type::get_safe_root)
            .def(
                    "add_child",
                    [](tree_type &tree, p_safe_node parent, p_safe_node child) {
                        tree.add_child(parent, child);
                    },
                    "parent"_a,
                    "child"_a);
    //            .def(
    //                    "remove_child",
    //                    [](tree_type &tree, safe_node_type parent, safe_node_type child) {
    //                        auto removed_child = tree.remove_child(parent->to_raw(), child->to_raw());
    //                        return removed_child
    //                                     ? std::make_shared<safe_node_type>(removed_child->get_data())
    //                                     : nullptr;
    //                    },
    //                    "parent"_a,
    //                    "child"_a)
    ////            .def(
    ////                    "add_child_mt",
    ////                    [](tree_type &tree, safe_node_type parent, safe_node_type child) {
    ////                        tree.add_child_mt(parent->to_raw(),
    ////                                          std::make_unique<raw_node_type>(child->get_data()));
    ////                    },
    ////                    "parent"_a,
    ////                    "child"_a)
    ////            .def(
    ////                    "remove_child_mt",
    ////                    [](tree_type &tree, safe_node_type parent, safe_node_type child) {
    ////                        auto removed_child
    ////                                = tree.remove_child_mt(parent->to_raw(), child->to_raw());
    ////                        return removed_child
    ////                                     ? std::make_shared<safe_node_type>(removed_child->get_data())
    ////                                     : nullptr;
    ////                    },
    ////                    "parent"_a,
    ////                    "child"_a)
    //            .DEF_IO_BINARY_METHODS(tree_type, false)
    //            .DEF_IO_JSON_METHODS(tree_type)
    //            .DEF_PY_BINARY_PICKLING(tree_type);

    using search_metrics_type = odin::SearchMetrics<Float>;
    py::class_<search_metrics_type>(m, ("SearchMetrics" + suffix).c_str())
            .def_readwrite("search_time", &search_metrics_type::search_time)
            .def_readwrite("iterations", &search_metrics_type::iterations)
            .def_readwrite("nodes_explored", &search_metrics_type::nodes_explored)
            .DEF_REPR(search_metrics_type)
            .DEF_IO_BINARY_METHODS(search_metrics_type, false)
            .DEF_IO_JSON_METHODS(search_metrics_type);

    using search_tracker_type = odin::SearchTracker<Float>;
    py::class_<search_tracker_type>(m, ("SearchTracker" + suffix).c_str())
            .def(py::init<>())
            .def("start", &search_tracker_type::start)
            .def("end", &search_tracker_type::end, "iters"_a, "nodes"_a);
}

#endif//BIND_TREE_HPP
