#ifndef BIND_LOGGING_HPP
#define BIND_LOGGING_HPP

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <odin/logging.hpp>

namespace py = pybind11;

#include <chrono>
#include <unordered_map>

std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timers;

auto start_timer
        = [](std::string name) { timers[name] = std::chrono::high_resolution_clock::now(); };

auto stop_timer = [](std::string name) {
    auto end_time   = std::chrono::high_resolution_clock::now();
    auto start_time = timers[name];
    timers.erase(name);

    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    SPDLOG_INFO("Timer {} stopped. Elapsed time: {} seconds.", name, elapsed_seconds.count());
};
#include <fmt/color.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#define DEFAULT_LOGGER_CREATOR spdlog::stdout_color_mt
#define DEFAULT_LOGGER_NAME "pydin"

// Helper function
template<typename T>
inline std::shared_ptr<T> get_or_create_logger(const std::string &name,
                                               spdlog::color_mode mode
                                               = spdlog::color_mode::automatic) {
    auto logger = spdlog::get(name);
    if (!logger) {
        logger = DEFAULT_LOGGER_CREATOR(name, mode);
        spdlog::register_logger(logger);
    }
    return logger;
}


void bind_logging(py::module &m, const std::string &suffix = "") {


    //    py::module fmt_m = m.def_submodule("fmt", "fmtlib submodule");
    //
    //    // Define color enumeration
    //    py::enum_<fmt::color>(fmt_m, "Color")
    //            .value("black", fmt::color::black)
    //            .value("red", fmt::color::red)
    //            .value("green", fmt::color::green)
    //            .value("yellow", fmt::color::yellow)
    //            .value("blue", fmt::color::blue)
    //            .value("magenta", fmt::color::magenta)
    //            .value("cyan", fmt::color::cyan)
    //            .value("white", fmt::color::white)
    //            .export_values();
    //
    //    // Define text_style class
    //    py::class_<fmt::text_style>(fmt_m, "TextStyle")
    //            .def(py::init<>())
    //            .def(py::init<const fmt::text_style &>())
    //            .def("__or__", &fmt::text_style::operator|)
    //            .def("get_foreground", &fmt::text_style::get_foreground)
    //            .def("get_background", &fmt::text_style::get_background)
    //            .def("has_emphasis", &fmt::text_style::has_emphasis)
    //            .def("get_emphasis", &fmt::text_style::get_emphasis)
    //            .def("set_emphasis", &fmt::text_style::set_emphasis)
    //            .def("unset_emphasis", &fmt::text_style::unset_emphasis)
    //            .def("get_intense", &fmt::text_style::get_intense);
    //    enum class pattern_time_type
    //    {
    //        local, // log localtime
    //        utc    // log utc
    //    };
    py::enum_<spdlog::pattern_time_type>(m, "pattern_time_type")
            .value("local", spdlog::pattern_time_type::local)
            .value("utc", spdlog::pattern_time_type::utc)
            .export_values();

    py::class_<spdlog::sinks::sink>(m, "sink");

    using namespace py::literals;
    using logger_type = spdlog::logger;
    using logger_type = spdlog::logger;
    py::class_<logger_type, std::shared_ptr<logger_type>>(m, "Logger")
            .def(py::init<std::string>())
            .def(
                    "set_level",
                    [](logger_type &self, spdlog::level::level_enum level) {
                        return self.set_level(level);
                    },
                    "level"_a)
            .def(
                    "set_pattern",
                    [](logger_type              &self,
                       std::string               pattern,
                       spdlog::pattern_time_type time_type) {
                        return self.set_pattern(pattern, time_type);
                    },
                    "pattern"_a,
                    "time_type"_a = spdlog::pattern_time_type::local)
            .def(
                    "trace",
                    [](logger_type &self, const std::string &msg) { return self.trace(msg); },
                    "msg"_a)
            .def(
                    "debug",
                    [](logger_type &self, const std::string &msg) { return self.debug(msg); },
                    "msg"_a)
            .def(
                    "info",
                    [](logger_type &self, const std::string &msg) { return self.info(msg); },
                    "msg"_a)
            .def(
                    "warn",
                    [](logger_type &self, const std::string &msg) { return self.warn(msg); },
                    "msg"_a)
            .def(
                    "error",
                    [](logger_type &self, const std::string &msg) { return self.error(msg); },
                    "msg"_a)
            .def(
                    "critical",
                    [](logger_type &self, const std::string &msg) { return self.critical(msg); },
                    "msg"_a)
            .def("should_log", &logger_type::should_log, "msg_level"_a)
            .def("should_backtrace", &logger_type::should_backtrace)
            .def("level", &logger_type::level)
            .def("name", &logger_type::name)
            .def("enable_backtrace", &logger_type::enable_backtrace, "n_messages"_a)
            .def("disable_backtrace", &logger_type::disable_backtrace)
            .def("dump_backtrace", &logger_type::dump_backtrace)
            .def("flush", &logger_type::flush)
            .def("flush_on", &logger_type::flush_on, "log_level"_a)
            .def("flush_level", &logger_type::flush_level)
            .def(
                    "sinks",
                    [](logger_type &l) { return l.sinks(); },
                    py::return_value_policy::reference_internal);

    py::enum_<spdlog::color_mode>(m, "color_mode")
            .value("always", spdlog::color_mode::always)
            .value("automatic", spdlog::color_mode::automatic)
            .value("never", spdlog::color_mode::never)
            .export_values();

    // Lambda wrapper for spdlog::get or create
//    m.def(
    //            "get_or_create_logger",
    //            [](const std::string &name, spdlog::color_mode mode) {
    //                auto logger = spdlog::get(name);
    //                if (logger == nullptr) {
    //                    logger = spdlog::stdout_color_mt(name, mode);
    //                    spdlog::register_logger(logger);
    //                }
    //                return logger;
    //            },
    //            "name"_a,
    //            "color_mode"_a = spdlog::color_mode::automatic);

    m.def(
            "stdout_color_mt",
            [](const std::string &name, spdlog::color_mode mode) {
                return spdlog::stdout_color_mt(name, mode);
            },
            "name"_a,
            "color_mode"_a = spdlog::color_mode::automatic);

    m.def(
            "stdout_color_st",
            [](const std::string &name, spdlog::color_mode mode) {
                return spdlog::stdout_color_st(name, mode);
            },
            "name"_a,
            "color_mode"_a = spdlog::color_mode::automatic);

    m.def("register_logger", &spdlog::register_logger);


    //    spdlog::stdout_color_mt("console");
    // Lambda wrapper for spdlog::get
    m.def("get_logger", &spdlog::get, "name"_a);// PEP8 alias
    m.def("getLogger", &spdlog::get, "name"_a); // Logging alias
    m.def("get", &spdlog::get, "name"_a);       // spdlog default


    py::enum_<spdlog::level::level_enum>(m, "LogLevel", py::arithmetic())
            .value("TRACE", spdlog::level::trace)
            .value("DEBUG", spdlog::level::debug)
            .value("INFO", spdlog::level::info)
            .value("WARN", spdlog::level::warn)
            .value("ERROR", spdlog::level::err)
            .value("CRITICAL", spdlog::level::critical)
            .export_values();

    m.def(
            "set_pattern",
            [](const std::string &pattern) { spdlog::set_pattern(pattern); },
            "pattern"_a);
    m.def(
            "set_level",
            [](int level) { spdlog::set_level(static_cast<spdlog::level::level_enum>(level)); },
            "level"_a);
    m.def("debug", static_cast<void (*)(const std::string &)>(&spdlog::debug), "msg"_a);
    m.def("info", static_cast<void (*)(const std::string &)>(&spdlog::info), "msg"_a);
    m.def("warn", static_cast<void (*)(const std::string &)>(&spdlog::warn), "msg"_a);
    m.def("error", static_cast<void (*)(const std::string &)>(&spdlog::error), "msg"_a);
    m.def("critical", static_cast<void (*)(const std::string &)>(&spdlog::critical), "msg"_a);
    m.def("trace", static_cast<void (*)(const std::string &)>(&spdlog::trace), "msg"_a);

    m.def("start_timer", start_timer, "Starts a timer with the given name", pybind11::arg("name"));
    m.def("stop_timer",
          stop_timer,
          "Stops a timer with the given name and logs the elapsed time",
          pybind11::arg("name"));

    py::class_<ConsoleLogger>(m, ("ConsoleLogger" + suffix).c_str())
            .def(py::init<>())
            .def("debug", &ConsoleLogger::debug<std::string>)
            .def("info", &ConsoleLogger::info<std::string>)
            .def("warn", &ConsoleLogger::warn<std::string>)
            .def("error", &ConsoleLogger::error<std::string>)
            .def("critical", &ConsoleLogger::critical<std::string>)
            .def("trace", &ConsoleLogger::trace<std::string>)
            .def("set_level", &ConsoleLogger::set_level);

    py::class_<FileLogger>(m, ("FileLogger" + suffix).c_str())
            .def(py::init<const std::string &>())
            .def("debug", &FileLogger::debug<std::string>)
            .def("info", &FileLogger::info<std::string>)
            .def("warn", &FileLogger::warn<std::string>)
            .def("error", &FileLogger::error<std::string>)
            .def("critical", &FileLogger::critical<std::string>)
            .def("trace", &FileLogger::trace<std::string>)
            .def("set_level", &FileLogger::set_level);

    m.def("global_console_logger", &global_console_logger, py::return_value_policy::reference);
    m.def("global_file_logger", &global_file_logger, py::return_value_policy::reference);

    py::class_<VLogger>(m, ("VLogger" + suffix).c_str())
            .def(py::init<int>())
            .def("set_verbosity", &VLogger::set_verbosity)
            .def("get_verbosity", &VLogger::get_verbosity);
}

#endif//BIND_LOGGING_HPP