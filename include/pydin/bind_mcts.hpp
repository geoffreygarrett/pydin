#ifndef BIND_MCTS_HPP
#define BIND_MCTS_HPP

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <odin/optimization/tree/mcts.hpp>
#include <pydin/details/mcts.hpp>


namespace state {
    template<typename Integer, typename Float>
    using mixed_integer_program = std::tuple <std::vector<Integer>, std::vector<Float>>;
}

namespace action {
    template<typename Integer = int>
    using single_integer = Integer;
}

namespace reward {
    template<typename Float = double>
    using single_scalar = Float;
}


// mixed integer programming state
//
//template<typename Integer, typename Float>
//void test_cpp(const MIPState<Integer, Float> &state) {
//    auto &int_vec  = std::get<0>(state);
//    auto &cont_vec = std::get<1>(state);
//
//    // Now you can use the Eigen vectors
//    // For example, print the size of the vectors:
//    std::cout << "int_size: " << int_vec.size() << std::endl;
//    std::cout << "cont_size: " << cont_vec.size() << std::endl;
//
//    // Or loop over the elements:
//    for (int i = 0; i < int_vec.size(); i++) {
//        std::cout << "integer[" << i << "]: " << int_vec[i] << std::endl;
//    }
//
//    for (int i = 0; i < cont_vec.size(); i++) {
//        std::cout << "continuous[" << i << "]: " << cont_vec[i] << std::endl;
//    }
//}

//template<typename Float>
//py bind_running_stats(py::module &m, const std::string &suffix) {
//
//}

template<typename State, typename Action, typename Reward, typename Float = double>
void bind_mcts(py::module &m, const std::string &suffix) {

    using RunningStatsType = RunningStats<double>;
    py::class_<RunningStatsType>(m, "RunningStats")
            .def(py::init<bool, bool, bool, bool>(),
                 "calc_mean"_a = false,
                 "calc_variance"_a = false,
                 "calc_min"_a = false,
                 "calc_max"_a = true)
            .def("push", &RunningStatsType::push, "value"_a)
            .def("mean", &RunningStatsType::mean)
            .def("variance", &RunningStatsType::variance)
            .def("standard_deviation", &RunningStatsType::standard_deviation)
            .def("min", &RunningStatsType::min)
            .def("max", &RunningStatsType::max)
            .def("count", &RunningStatsType::count)
            .def("clear", &RunningStatsType::clear);

    //    m.def("test_cpp", &test_cpp<int, double>,
    //          "Test function for MIPState<int, Float>",
    //          py::arg_v("state", nullptr, "The MIPState object to work on").noconvert());

    using py_node_type = PyNode<State, Action, Reward>;
    using py_node_sptr = std::shared_ptr<py_node_type>;
//    py::module::import("core.RunningStats");

    auto node_repr = [](const py_node_type &self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
    };

    auto get_children = [](py_node_sptr self) {
        std::vector <py_node_sptr> child_nodes;
        std::transform(self->children.begin(), self->children.end(), std::back_inserter(child_nodes),
                       [](auto &child) { return std::make_shared<py_node_type>(*child); });
        return child_nodes;
    };

    auto get_parent = [](py_node_sptr self) -> py::object {
        if (auto parent = self->parent.lock()) {
            return py::cast(std::make_shared<py_node_type>(*parent), py::return_value_policy::reference_internal);
        } else {
            return py::none();
        }
    };
    //    explicit RunningStats(bool calc_mean = false, bool calc_variance = false, bool calc_min = false, bool calc_max = true)


    py::class_<py_node_type, py_node_sptr>(m, ("PyNode" + suffix).c_str())
            .def("__repr__", node_repr)
            .def("get_children", get_children, py::return_value_policy::reference)
            .def("get_parent", get_parent)
            .def_readonly("is_leaf", &py_node_type::is_leaf)
            .def_readonly("state", &py_node_type::state)
            .def_readonly("action", &py_node_type::action)
            .def_readonly("reward_stats", &py_node_type::reward_stats)
            .def_readonly("is_terminal", &py_node_type::is_terminal);


    using selection_policy = typename MCTS<State, Action, Reward, Float>::selection_policy;
    using state_transition = typename MCTS<State, Action, Reward, Float>::state_transition;
    using action_generator = typename MCTS<State, Action, Reward, Float>::action_generator;
    using is_terminal = typename MCTS<State, Action, Reward, Float>::is_terminal;
    using reward = typename MCTS<State, Action, Reward, Float>::reward;
    using node_type = Node<State, Action, Reward>;

    // Bind the CppNode class
    py::class_<node_type>(m, ("CppNode" + suffix).c_str())
            .def_readonly("reward_stats", &node_type::reward_stats);

    using value_estimator_type = typename SelectionPolicy<node_type, Float>::value_estimator_type;

    // Create a default_value_estimator
    value_estimator_type max_value_estimator = [](const node_type *node) {
        return node->reward_stats.max();
    };

    value_estimator_type mean_value_estimator = [](const node_type *node) {
        return node->reward_stats.mean();
    };

    value_estimator_type min_value_estimator = [](const node_type *node) {
        return node->reward_stats.min();
    };

    m.attr("min_value_estimator") = py::cast(max_value_estimator);
    m.attr("mean_value_estimator") = py::cast(mean_value_estimator);
    m.attr("max_value_estimator") = py::cast(min_value_estimator);

    // Bind the SelectionPolicy class. The "value" is a pure virtual function, and so is the destructor.
    py::class_ < SelectionPolicy < node_type, Float >, std::shared_ptr < SelectionPolicy < node_type, Float>>>(m, (
            "SelectionPolicy" + suffix).c_str())
            .def("__repr__", [](const SelectionPolicy <node_type, Float> &self) {
                std::stringstream ss;
                ss << "SelectionPolicy()";
                return ss.str();
            });

    py::class_ < UCB1 < node_type, Float >, SelectionPolicy < node_type, Float >,
            std::shared_ptr < UCB1 < node_type, Float>>>(m, ("UCB1" + suffix).c_str())
            .def(
                    py::init<Float, value_estimator_type>(),
                    "cp"_a = 0.01,
                    py::arg_v("value_estimator", max_value_estimator, "max_value_estimator()"))
            .def("__repr__", [](const UCB1 <node_type, Float> &self) {
                std::stringstream ss;
                ss << "UCB1(cp=" << self.get_cp() << ")";
                return ss.str();
            });

    py::class_ < UCB1Tuned < node_type, Float >, SelectionPolicy < node_type, Float >,
            std::shared_ptr < UCB1Tuned < node_type, Float>>>(m, ("UCB1Tuned" + suffix).c_str())
            .def(
                    py::init<Float, value_estimator_type>(),
                    "cp"_a = 0.01,
                    py::arg_v("value_estimator", max_value_estimator, "max_value_estimator()"))
            .def("__repr__", [](const UCB1Tuned <node_type, Float> &self) {
                std::stringstream ss;
                ss << "UCB1Tuned(cp=" << self.get_cp() << ")";
                return ss.str();
            });

    py::class_ < EpsilonGreedy < node_type, Float >, SelectionPolicy < node_type, Float >,
            std::shared_ptr < EpsilonGreedy < node_type, Float>>>(m, ("EpsilonGreedy" + suffix).c_str())
            .def(
                    py::init<Float, value_estimator_type>(),
                    "epsilon"_a = 0.01,
                    py::arg_v("value_estimator", max_value_estimator, "max_value_estimator()"))
            .def("__repr__", [](const EpsilonGreedy <node_type, Float> &self) {
                std::stringstream ss;
                ss << "EpsilonGreedy(epsilon=" << self.get_epsilon() << ")";
                return ss.str();
            });

    using mcts_type = MCTS<State, Action, Reward, Float>;
    py::class_ < mcts_type, std::unique_ptr < mcts_type >> (m, ("MCTS" + suffix).c_str())
            .def(py::init < State,
                 action_generator,
                 state_transition,
                 is_terminal,
                 reward,
                 selection_policy,
                 size_t,
                 std::optional < Action >> (),
                 R"doc(
                 Construct a new MCTS object

                 Parameters
                 ----------
                 initial_state : State
                     The initial state of the MCTS.
                 initial_action : Action
                     The initial action of the MCTS.
                 action_generator : function
                     A function that generates the possible actions for a given state.
                 transition : function
                     A function that takes a state and action, and returns a new state.
                 is_terminal : function
                     A function that checks whether a given state is terminal.
                 reward : function
                     A function that calculates the reward for a given state.
                 selection_policy : function, optional
                     A function that determines the selection policy, by default UBC1.
                 seed : int, optional
                     A seed for the random number generator, by default random.

                 Returns
                 -------
                 MCTS
                     An initialized MCTS object.
                 )doc",
                 "initial_state"_a,
                 "action_generator"_a,
                 "transition"_a,
                 "is_terminal"_a,
                 "reward"_a,
                 "selection_policy"_a = std::make_shared < UCB1 < node_type, Float >> (1.0, max_value_estimator),
                 "seed"_a = std::random_device()(),
                 "initial_action"_a = std::nullopt)
            .def(
                    "__repr__", [](const mcts_type &mcts) {
                        std::stringstream ss;
                        ss << mcts;
                        return ss.str();
                    },
                    R"doc(
                    Convert the MCTS object to a string representation.

                    Returns
                    -------
                    str
                        The string representation of the MCTS object.
                    )doc")
            .def("search", &mcts_type::search, "iterations"_a = 1000, "seconds"_a = -1.0, "expand_all"_a = false,
                 "contraction"_a = true,
                 R"doc(
                Perform a search with the given number of iterations.

                Parameters
                ----------
                iterations : int, optional
                    The number of iterations for the search, by default 1000.
                seconds : float, optional
                    The number of seconds for the search, by default -1.0.
                expand_all : bool, optional
                    Whether to expand all nodes, by default False.

                Returns
                -------
                None
                )doc")
            .def(
                    "get_root", [](mcts_type &self) {
                        auto root = self.get_root();
                        // Convert the C++ Node to a Python PyNode using pre-defined function
                        return convert_node_to_py_node(*root);
                    },
                    R"doc(
                Get the root of the MCTS.

                Returns
                -------
                Node
                    The root of the MCTS.
                )doc");
    //            .def("search", &mcts_type::search, "iterations"_a = 1000)
    //                 "num_iterations"_a = 1000,
    //                 "num_threads"_a    = 1,
    //                 "max_depth"_a      = -1,
    //                 "verbose"_a        = false)

    //            .def("get_root", &mcts_type::get_root);
    //            .def("search", &MCTS<State, Action, Reward, Float>::search,
    //                 "num_iterations"_a = 1000
    //                 //                 "num_threads"_a    = 1,
    //                 //                 "max_depth"_a      = -1,
    //                 //                 "verbose"_a        = false
    //                 )
    //            .def("get_best_trajectory", &MCTS<State, Action, Reward, Float>::get_best_trajectory);

    // OPTIMIZATION TOOLS
    //    using NodeVec = Node<State, Action, Float>;
    //    py::class_<NodeVec>(m, TYPE_SUFFIX("MCTSNode"))
    //            .def(py::init<>(), "init_default")
    //            .def(py::init<std::shared_ptr<NodeVec>, State, Action, bool>(), "init_with_args",
    //                 "parent"_a,
    //                 "state"_a,
    //                 "action"_a,
    //                 "move"_a)
    //            .def_readonly("total_score", &NodeVec::total_score)
    //            .def_readonly("visit_count", &NodeVec::visit_count)
    //            .def_readonly("move", &NodeVec::move)
    //            .def_readonly("state", &NodeVec::state)
    //            .def_readonly("action", &NodeVec::action)
    //            .def_readonly("is_terminal", &NodeVec::is_terminal)
    //            //                    void print(const std::string &prefix = "", bool is_tail = true, int max_depth = -1, bool best_branch_only = false) const {
    //            .def("print", &NodeVec::print,
    //                 "prefix"_a           = "",
    //                 "is_tail"_a          = true,
    //                 "max_depth"_a        = -1,
    //                 "best_branch_only"_a = false,
    //                 "is_root"_a          = true);
    //    //            .def("get_children", &get_children<Float>);

    //    using MCTSType = MCTS<State, Action, Float>;
    //    py::class_<MCTSType>(m, TYPE_SUFFIX("MCTS"))
    //            .def(py::init<State, Action,
    //                          std::function<std::vector<Action>(State)>,
    //                          std::function<State(State, Action)>,
    //                          std::function<bool(State)>,
    //                          std::function<Float(State, bool)>>(),
    //                 "initial_state"_a,
    //                 "initial_action"_a,
    //                 "get_actions"_a,
    //                 "transition"_a,
    //                 "is_terminal"_a,
    //                 "reward"_a)
    //            .def("best_action", &MCTSType::best_action)
    //            .def("print_tree", &MCTSType::print_tree,
    //                 "max_depth"_a        = -1,
    //                 "best_branch_only"_a = false)
    //            .def("execute", &MCTSType::execute,
    //                 "max_iter"_a = 1000);
}


#endif// BIND_MCTS_HPP