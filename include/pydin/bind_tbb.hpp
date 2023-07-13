#ifndef BIND_TBB_HPP
#define BIND_TBB_HPP

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <thread>

#include <tbb/global_control.h>

namespace py = pybind11;

class TBBControl {
private:
    std::unique_ptr<tbb::global_control> control;

public:
    TBBControl(std::size_t num_threads = std::thread::hardware_concurrency()) {
        set_max_allowed_parallelism(num_threads);
    }

    ~TBBControl() = default;

    void set_max_allowed_parallelism(std::size_t num_threads) {
        control = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism,
                num_threads);
    }

    std::size_t get_max_allowed_parallelism() const {
        return tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
    }

    std::size_t get_thread_stack_size() const {
        return tbb::global_control::active_value(tbb::global_control::thread_stack_size);
    }
};

void bind_tbb(py::module &m) {
    py::class_<TBBControl>(m, "TBBControl")
            .def(py::init<>(), "Initialize a new TBB control.")
            .def("__enter__", [](TBBControl &self) -> TBBControl & { return self; })
            .def("__exit__", [](TBBControl &self, py::args) {})
            .def_property("max_allowed_parallelism",
                          &TBBControl::get_max_allowed_parallelism,
                          &TBBControl::set_max_allowed_parallelism,
                          "Get or set the maximum number of threads TBB is allowed to utilize.")
            .def("get_thread_stack_size",
                 &TBBControl::get_thread_stack_size,
                 "Get the stack size of the threads TBB creates.");

    m.def("hardware_concurrency",
          &std::thread::hardware_concurrency,
          "Get the number of hardware threads available.");
}

#endif//BIND_TBB_HPP
