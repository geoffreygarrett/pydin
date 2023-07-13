
#ifndef PYDIN_INCLUDE_COMMON_HPP
#define PYDIN_INCLUDE_COMMON_HPP

#include <pybind11/pybind11.h>

#include <fstream>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/access.hpp>

// define macro for suffixing function names
#define TYPE_SUFFIX(name) (name + suffix).c_str()


template<typename Func, typename... Args>
auto execute_with_gil_release(Func func, Args... args) {
    pybind11::gil_scoped_release release;
    auto                         result = func(std::forward<Args>(args)...);
    pybind11::gil_scoped_acquire acquire;
    return result;
}

#define DECLARE_REPR(cls)                                                                         \
    def("__repr__", [](const cls &obj) {                                                          \
        std::ostringstream os;                                                                    \
        os << obj;                                                                                \
        return os.str();                                                                          \
    })

//Explicit is better than implicit
#define DECLARE_TO_JSON(cls)                                                                      \
    def("to_json", [](const cls &object, const std::string &file = "") {                          \
        std::ostringstream os;                                                                    \
        {                                                                                         \
            cereal::JSONOutputArchive archive(os);                                                \
            archive(cereal::make_nvp("data", object));                                            \
        }                                                                                         \
        if (!file.empty()) {                                                                      \
            std::ofstream ofs(file);                                                              \
            if (ofs) {                                                                            \
                ofs << os.str();                                                                  \
                ofs.close();                                                                      \
            } else {                                                                              \
                throw std::runtime_error("Failed to open the file: " + file);                     \
            }                                                                                     \
        }                                                                                         \
        return os.str();                                                                          \
    })

#define DECLARE_FROM_JSON(cls)                                                                    \
    def_static(                                                                                   \
            "from_json",                                                                          \
            [](const std::string &json_str) {                                                     \
                std::istringstream is(json_str);                                                  \
                cls                object;                                                        \
                {                                                                                 \
                    cereal::JSONInputArchive archive(is);                                         \
                    archive(cereal::make_nvp("data", object));                                    \
                }                                                                                 \
                return object;                                                                    \
            },                                                                                    \
            py::arg("json_str"),                                                                  \
            "Deserialize an instance of the class from a JSON string.")

#define DECLARE_LOAD_JSON(cls)                                                                    \
    def_static(                                                                                   \
            "load_json",                                                                          \
            [](const std::string &file) {                                                         \
                cls           object;                                                             \
                std::ifstream ifs(file);                                                          \
                if (!ifs) { throw std::runtime_error("Failed to open the file: " + file); }       \
                {                                                                                 \
                    cereal::JSONInputArchive archive(ifs);                                        \
                    archive(cereal::make_nvp("data", object));                                    \
                }                                                                                 \
                return object;                                                                    \
            },                                                                                    \
            py::arg("filepath"),                                                                  \
            "Load an instance of the class from a JSON file.")

#define DECLARE_TO_BINARY(cls)                                                                    \
    def(                                                                                          \
            "to_binary",                                                                          \
            [](const cls &object, const std::string &file = "", bool portable = false) {          \
                std::stringstream ss;                                                             \
                if (portable) {                                                                   \
                    cereal::PortableBinaryOutputArchive archive(ss);                              \
                    archive(cereal::make_nvp("data", object));                                    \
                } else {                                                                          \
                    cereal::BinaryOutputArchive archive(ss);                                      \
                    archive(cereal::make_nvp("data", object));                                    \
                }                                                                                 \
                if (!file.empty()) {                                                              \
                    std::ofstream ofs(file, std::ios::binary);                                    \
                    if (ofs) {                                                                    \
                        ofs << ss.rdbuf();                                                        \
                        ofs.close();                                                              \
                    } else {                                                                      \
                        throw std::runtime_error("Failed to open the file: " + file);             \
                    }                                                                             \
                }                                                                                 \
                std::string s = ss.str();                                                         \
                return py::bytes(s);                                                              \
            },                                                                                    \
            py::arg("file")     = "",                                                             \
            py::arg("portable") = false)

#define DECLARE_FROM_BINARY(cls)                                                                  \
    def_static(                                                                                   \
            "from_binary",                                                                        \
            [](const std::string &bin_str, bool portable = false) {                               \
                std::istringstream is(bin_str, std::ios::binary);                                 \
                cls                object;                                                        \
                if (portable) {                                                                   \
                    cereal::PortableBinaryInputArchive archive(is);                               \
                    archive(cereal::make_nvp("data", object));                                    \
                } else {                                                                          \
                    cereal::BinaryInputArchive archive(is);                                       \
                    archive(cereal::make_nvp("data", object));                                    \
                }                                                                                 \
                return object;                                                                    \
            },                                                                                    \
            py::arg("bin_str"),                                                                   \
            py::arg("portable") = false)

#define DECLARE_LOAD_BINARY(cls)                                                                  \
    def_static(                                                                                   \
            "load_binary",                                                                        \
            [](const std::string &file, bool portable = false) {                                  \
                cls           object;                                                             \
                std::ifstream ifs(file, std::ios::binary);                                        \
                if (!ifs) { throw std::runtime_error("Failed to open the file: " + file); }       \
                if (portable) {                                                                   \
                    cereal::PortableBinaryInputArchive archive(ifs);                              \
                    archive(cereal::make_nvp("data", object));                                    \
                } else {                                                                          \
                    cereal::BinaryInputArchive archive(ifs);                                      \
                    archive(cereal::make_nvp("data", object));                                    \
                }                                                                                 \
                return object;                                                                    \
            },                                                                                    \
            py::arg("filepath"),                                                                  \
            py::arg("portable") = false)


#define DECLARE_BINARY_PICKLING(cls)                                                                \
    def(py::pickle(                                                                                 \
            [](const cls &obj) {                                                                    \
                std::stringstream                   ss;                                             \
                cereal::PortableBinaryOutputArchive archive(ss);                                    \
                archive(cereal::make_nvp("data", obj));                                             \
                std::string s = ss.str();                                                           \
                return py::bytes(s);                                                                \
            },                                                                                      \
            [](const py::bytes &state) {                                                            \
                std::istringstream                 is(state.cast<std::string>(), std::ios::binary); \
                cls                                object;                                          \
                cereal::PortableBinaryInputArchive archive(is);                                     \
                archive(cereal::make_nvp("data", object));                                          \
                return object;                                                                      \
            }))

#define DECLARE_IO_FUNCTIONS(cls)                                                                 \
    DECLARE_TO_JSON(cls)                                                                          \
            .DECLARE_FROM_JSON(cls)                                                               \
            .DECLARE_LOAD_JSON(cls)                                                               \
            .DECLARE_TO_BINARY(cls)                                                               \
            .DECLARE_FROM_BINARY(cls)                                                             \
            .DECLARE_LOAD_BINARY(cls)

#endif// PYDIN_INCLUDE_COMMON_HPP