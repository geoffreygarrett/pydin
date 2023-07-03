#ifndef BIND_LOGGING_HPP
#define BIND_LOGGING_HPP

#include <odin/logging.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

// C++ function that logs a message
void cxx_log(std::string level, std::string msg) {
    if (level == "INFO") {
        ODIN_LOG_INFO << msg;
    } else if (level == "WARNING") {
        ODIN_LOG_WARNING << msg;
    } else if (level == "ERROR") {
        ODIN_LOG_ERROR << msg;
    } else if (level == "FATAL") {
        ODIN_LOG_FATAL << msg;
    }
};

void bind_logging(py::module &m, const std::string &suffix = "") {
    using namespace pybind11::literals;
    namespace py = pybind11;
    //    m_logging.attr("INFO")    = py::int_(INFO);
    //    m_logging.attr("WARNING") = py::int_(WARNING);
    //    m_logging.attr("ERROR")   = py::int_(ERROR);
    //    m_logging.attr("FATAL")   = py::int_(FATAL);
    INIT_ODIN_LOGGING("pydin", "./log/pydin.log");

    m.def(
            "set_log_destination", [](const int level, const char *destination) {
                ODIN_SET_LOG_DESTINATION(level, destination);
            },
            "Set the destination of the logging system", "level"_a, "destination"_a);

    m.def(
            "log_info", [](const std::string &msg) {
                ODIN_LOG_INFO << msg;
            },
            "Log an info message", "msg"_a);

    m.def(
            "log_warning", [](const std::string &msg) {
                ODIN_LOG_WARNING << msg;
            },
            "Log a warning message", "msg"_a);

    m.def(
            "log_error", [](const std::string &msg) {
                ODIN_LOG_ERROR << msg;
            },
            "Log an error message", "msg"_a);

    m.def(
            "log_fatal", [](const std::string &msg) {
                ODIN_LOG_FATAL << msg;
            },
            "Log a fatal message", "msg"_a);

    m.def(
            "set_vlog_level", [](int level) {
                ODIN_SET_VLOG_LEVEL(level);
            },
            "Set the verbosity level of the logging system", "level"_a);

    m.def(
            "vlog", [](int level, const std::string &msg) {
                ODIN_VLOG(level) << msg;
            },
            "Log a message with a verbosity level", "level"_a, "msg"_a);


    // Register it with the Python logger
    py::module logging      = py::module::import("logging");
    logging.attr("info")    = py::cpp_function([](std::string msg) { cxx_log("INFO", std::move(msg)); });
    logging.attr("warning") = py::cpp_function([](std::string msg) { cxx_log("WARNING", std::move(msg)); });
    logging.attr("error")   = py::cpp_function([](std::string msg) { cxx_log("ERROR", std::move(msg)); });
    logging.attr("fatal")   = py::cpp_function([](std::string msg) { cxx_log("FATAL", std::move(msg)); });
}

#endif// BIND_LOGGING_HPP