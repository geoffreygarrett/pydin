#ifndef BIND_MCTS_HPP
#define BIND_MCTS_HPP

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <odin/optimization/tree/mcts.hpp>
//#include <odin/optimization/tree/mcts_serialization.hpp>
#include <pydin/common.hpp>
#include <pydin/details/mcts.hpp>
#include <pydin/docstrings.hpp>


namespace state {
    template<typename Integer, typename Float>
    using mixed_integer_program = std::tuple<std::vector<Integer>, std::vector<Float>>;
}

namespace action {
    template<typename Integer = int>
    using single_integer = Integer;
}

namespace reward {
    template<typename Float = double>
    using single_scalar = Float;
}

//
//template<typename State, typename Action, typename Reward>
//std::unique_ptr<Node<State, Action, Reward>>
//deep_copy_node(const Node<State, Action, Reward> &node) {
//    auto copied_node   = std::make_unique<Node<State, Action, Reward>>();
//    copied_node->state = node.state;
//    copied_node->parent
//            = nullptr;// the parent will be set when the child is added to the parent node
//    copied_node->children.clear();// ensure the children vector is empty
//    for (const auto &child: node.children) {
//        auto copied_child    = deep_copy_node(*child);
//        copied_child->parent = copied_node.get();// set the parent pointer
//        copied_node->children.push_back(std::move(copied_child));
//    }
//    return copied_node;
//}

template<typename State, typename Action, typename Reward, typename Float = double>
void bind_mcts(py::module &m, const std::string &suffix = "") {

    using running_stats_type = RunningStats<Float>;
    py::class_<running_stats_type>(m, "RunningStats")
            .def(py::init<bool, bool, bool, bool>(),
                 "calc_mean"_a     = false,
                 "calc_variance"_a = false,
                 "calc_min"_a      = false,
                 "calc_max"_a      = true)
            .def("push", &running_stats_type::push, "value"_a)
            .def("mean", &running_stats_type::mean)
            .def("variance", &running_stats_type::variance)
            .def("standard_deviation", &running_stats_type::standard_deviation)
            .def("min", &running_stats_type::min)
            .def("max", &running_stats_type::max)
            .def("count", &running_stats_type::count)
            .def("clear", &running_stats_type::clear)
            .DECLARE_REPR(running_stats_type)
            .DECLARE_BINARY_PICKLING(running_stats_type)
            .DECLARE_IO_FUNCTIONS(running_stats_type);

    using py_node_type = PyNode<State, Action, Reward>;
    py::class_<py_node_type, std::shared_ptr<py_node_type>>(m, ("PyNode" + suffix).c_str())
            .def(py::init<const State, const std::optional<Action>>(),
                 "state"_a,
                 "action"_a = std::nullopt)
            .def("get_children",
                 [](const py_node_type &self) -> py::object { return py::cast(self.children); })
            .def("get_parent",
                 [](py_node_type &self) -> py::object {
                     return self.parent.lock() ? py::cast(*self.parent.lock()) : py::none();
                 })
            .def_readonly("is_leaf", &py_node_type::is_leaf)
            .def_readonly("state", &py_node_type::state)
            .def_readonly("action", &py_node_type::action)
            .def_readonly("reward_stats", &py_node_type::reward_stats)
            .def_readonly("is_terminal", &py_node_type::is_terminal)
            .DECLARE_REPR(py_node_type)
            .DECLARE_BINARY_PICKLING(py_node_type)
            .DECLARE_IO_FUNCTIONS(py_node_type);

    using node_type = Node<State, Action, Reward>;

    //    py::class_<node_type, std::shared_ptr<node_type>>(m, ("Node" + suffix).c_str())
    //            .def(py::init<const State, const std::optional<Action>>(),
    //                 "state"_a,
    //                 "action"_a = std::nullopt)
    //            .def("get_children",
    //                 [](const node_type &self) -> py::object {
    //                     std::vector<std::unique_ptr<node_type>> copied_children;
    //                     for (const auto &child: self.children) {
    //                         copied_children.push_back(deep_copy_node(*child));
    //                     }
    //                     return py::cast(copied_children);
    //                 })
    //            .def("get_parent",
    //                 [](node_type &self) -> py::object {
    //                     if (self.parent != nullptr) {
    //                         return py::cast(deep_copy_node(*self.parent));
    //                     } else {
    //                         return py::none();
    //                     }
    //                 })
    //            .def_readonly("is_leaf", &node_type::is_leaf)
    //            .def_readonly("state", &node_type::state)
    //            .def_readonly("action", &node_type::action)
    //            .def_readonly("reward_stats", &node_type::reward_stats)
    //            .def_readonly("is_terminal", &node_type::is_terminal);
    //            .DECLARE_REPR(node_type)
    //            .DECLARE_BINARY_PICKLING(node_type)
    //            .DECLARE_IO_FUNCTIONS(node_type);

    using selection_policy = typename MCTS<State, Action, Reward, Float>::fn_selection_policy;
    using state_transition = typename MCTS<State, Action, Reward, Float>::fn_state_transition;
    using action_generator = typename MCTS<State, Action, Reward, Float>::fn_action_generator;
    using is_terminal      = typename MCTS<State, Action, Reward, Float>::fn_is_terminal;
    using reward           = typename MCTS<State, Action, Reward, Float>::fn_reward;
    using node_type        = Node<State, Action, Reward>;

    //    // Bind the CppNode class
    //    py::class_<node_type>(m, ("CppNode" + suffix).c_str())
    //            .def_readonly("reward_stats", &node_type::reward_stats);

    using value_estimator_type = typename SelectionPolicy<node_type, Float>::value_estimator_type;

    // Create a default_value_estimator
    value_estimator_type max_value_estimator
            = [](const node_type *node) { return node->reward_stats.max(); };

    value_estimator_type mean_value_estimator
            = [](const node_type *node) { return node->reward_stats.mean(); };

    value_estimator_type min_value_estimator
            = [](const node_type *node) { return node->reward_stats.min(); };

    m.attr("min_value_estimator")  = py::cast(max_value_estimator);
    m.attr("mean_value_estimator") = py::cast(mean_value_estimator);
    m.attr("max_value_estimator")  = py::cast(min_value_estimator);

    // Bind the SelectionPolicy class. The "value" is a pure virtual function, and so is the destructor.
    py::class_<SelectionPolicy<node_type, Float>,
               std::shared_ptr<SelectionPolicy<node_type, Float>>>(
            m,
            ("SelectionPolicy" + suffix).c_str())
            .def("__repr__", [](const SelectionPolicy<node_type, Float> &self) {
                std::stringstream ss;
                ss << "SelectionPolicy()";
                return ss.str();
            });

    py::class_<UCB1<node_type, Float>,
               SelectionPolicy<node_type, Float>,
               std::shared_ptr<UCB1<node_type, Float>>>(m, ("UCB1" + suffix).c_str())
            .def(py::init<Float, value_estimator_type>(),
                 "cp"_a = 0.01,
                 py::arg_v("value_estimator", max_value_estimator, "max_value_estimator()"))
            .def("__repr__", [](const UCB1<node_type, Float> &self) {
                std::stringstream ss;
                ss << "UCB1(cp=" << self.get_cp() << ")";
                return ss.str();
            });

    py::class_<UCB1Tuned<node_type, Float>,
               SelectionPolicy<node_type, Float>,
               std::shared_ptr<UCB1Tuned<node_type, Float>>>(m, ("UCB1Tuned" + suffix).c_str())
            .def(py::init<Float, value_estimator_type>(),
                 "cp"_a = 0.01,
                 py::arg_v("value_estimator", max_value_estimator, "max_value_estimator()"))
            .def("__repr__", [](const UCB1Tuned<node_type, Float> &self) {
                std::stringstream ss;
                ss << "UCB1Tuned(cp=" << self.get_cp() << ")";
                return ss.str();
            });

    py::class_<EpsilonGreedy<node_type, Float>,
               SelectionPolicy<node_type, Float>,
               std::shared_ptr<EpsilonGreedy<node_type, Float>>>(
            m,
            ("EpsilonGreedy" + suffix).c_str())
            .def(py::init<Float, value_estimator_type>(),
                 "epsilon"_a = 0.01,
                 py::arg_v("value_estimator", max_value_estimator, "max_value_estimator()"))
            .def("__repr__", [](const EpsilonGreedy<node_type, Float> &self) {
                std::stringstream ss;
                ss << "EpsilonGreedy(epsilon=" << self.get_epsilon() << ")";
                return ss.str();
            });

    py::class_<SearchMetrics>(m, "SearchMetrics")
            .def_readonly("search_seconds", &SearchMetrics::search_seconds)
            .def_readonly("search_iterations", &SearchMetrics::search_iterations)
            .def_readonly("fevals_transitions_evaluated",
                          &SearchMetrics::fevals_transitions_evaluated)
            .def_readonly("fevals_rewards_evaluated", &SearchMetrics::fevals_rewards_evaluated)
            .def_readonly("fevals_actions_generated", &SearchMetrics::fevals_actions_generated)
            .def_readonly("fevals_terminal_checks", &SearchMetrics::fevals_terminal_checks)
            .def_readonly("fevals_selection_policy", &SearchMetrics::fevals_selection_policy)
            .DECLARE_REPR(SearchMetrics)
            .DECLARE_BINARY_PICKLING(SearchMetrics)
            .DECLARE_IO_FUNCTIONS(SearchMetrics);

    using mcts_type = MCTS<State, Action, Reward, Float>;
    py::class_<mcts_type, std::shared_ptr<mcts_type>>(m, ("MCTS" + suffix).c_str())
//            .def(py::init<std::shared_ptr<py_node_type>>(), "root"_a)
            .def(py::init<State,
                          action_generator,
                          state_transition,
                          is_terminal,
                          reward,
                          selection_policy,
                          size_t,
                          std::optional<Action>>(),
                 mcts_docstrings.c_str(),
                 "initial_state"_a,
                 "action_generator"_a,
                 "transition"_a,
                 "is_terminal"_a,
                 "reward"_a,
                 "selection_policy"_a
                 = std::make_shared<UCB1<node_type, Float>>(1.0, max_value_estimator),
                 "seed"_a           = std::random_device()(),
                 "initial_action"_a = std::nullopt)
            .DECLARE_REPR(mcts_type)
            //            .DECLARE_BINARY_PICKLING(mcts_type)
            //            .DECLARE_IO_FUNCTIONS(mcts_type)
            .def("set_fn_get_actions", &mcts_type::set_fn_get_actions)
            .def("set_fn_transition", &mcts_type::set_fn_transition)
            .def("set_fn_is_terminal", &mcts_type::set_fn_is_terminal)
            .def("set_fn_reward", &mcts_type::set_fn_reward)
            .def("set_fn_selection_policy", &mcts_type::set_fn_selection_policy)
            .def(
                    "search",
                    [&](mcts_type &self,
                        size_t     iterations,
                        double     seconds,
                        bool       expand_all,
                        bool       contraction,
                        int        leaf_parallelism) {
                        return execute_with_gil_release([&]() {
                            return self.search(iterations,
                                               seconds,
                                               expand_all,
                                               contraction,
                                               leaf_parallelism);
                        });
                    },
                    "iterations"_a       = 1000,
                    "seconds"_a          = -1.0,
                    "expand_all"_a       = false,
                    "contraction"_a      = true,
                    "leaf_parallelism"_a = 4,
                    mcts_search_docstrings.c_str())
            .def(
                    "get_root",
                    [](mcts_type &self) {
                        auto root = self.get_root();
                        // Convert the C++ Node to a Python PyNode using pre-defined function
                        return convert_node_to_py_node(*root);
                    },
                    mcts_get_root_docstrings.c_str());
}


#endif// BIND_MCTS_HPP