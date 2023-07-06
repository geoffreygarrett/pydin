#ifndef BIND_LOGGING_HPP
#define BIND_LOGGING_HPP

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <odin/logging.hpp>

namespace py = pybind11;

void bind_logging(py::module &m, const std::string &suffix = "") {
    using namespace py::literals;

    py::enum_<spdlog::level::level_enum>(m, "LogLevel", py::arithmetic())
            .value("TRACE", spdlog::level::trace)
            .value("DEBUG", spdlog::level::debug)
            .value("INFO", spdlog::level::info)
            .value("WARN", spdlog::level::warn)
            .value("ERROR", spdlog::level::err)
            .value("CRITICAL", spdlog::level::critical)
            .export_values();

    py::class_<ConsoleLogger>(m, ("ConsoleLogger" + suffix).c_str())
            .def(py::init<>())
            .def("debug", &ConsoleLogger::debug < std::string > )
            .def("info", &ConsoleLogger::info < std::string > )
            .def("warn", &ConsoleLogger::warn < std::string > )
            .def("error", &ConsoleLogger::error < std::string > )
            .def("critical", &ConsoleLogger::critical < std::string > )
            .def("trace", &ConsoleLogger::trace < std::string > )
            .def("set_level", &ConsoleLogger::set_level);

    py::class_<FileLogger>(m, ("FileLogger" + suffix).c_str())
            .def(py::init<const std::string &>())
            .def("debug", &FileLogger::debug < std::string > )
            .def("info", &FileLogger::info < std::string > )
            .def("warn", &FileLogger::warn < std::string > )
            .def("error", &FileLogger::error < std::string > )
            .def("critical", &FileLogger::critical < std::string > )
            .def("trace", &FileLogger::trace < std::string > )
            .def("set_level", &FileLogger::set_level);

    m.def("global_console_logger", &global_console_logger, py::return_value_policy::reference);
    m.def("global_file_logger", &global_file_logger, py::return_value_policy::reference);

    py::class_<VLogger>(m, ("VLogger" + suffix).c_str())
            .def(py::init<int>())
            .def("set_verbosity", &VLogger::set_verbosity)
            .def("get_verbosity", &VLogger::get_verbosity);
}

#endif //BIND_LOGGING_HPP