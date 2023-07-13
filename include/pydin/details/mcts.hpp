#ifndef DETAILS_MCTS_HPP
#define DETAILS_MCTS_HPP
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cereal/access.hpp>
#include <odin/optimization/tree/mcts.hpp>
namespace py = pybind11;
using namespace pybind11::literals;


template<typename State, typename Action, typename Reward>
struct PyNode : NodeBase<PyNode<State, Action, Reward>, State, Action, Reward>,
                public std::enable_shared_from_this<PyNode<State, Action, Reward>> {
    using NodeBase<PyNode, State, Action, Reward>::NodeBase;
    std::weak_ptr<PyNode<State, Action, Reward>>                parent;
    std::vector<std::shared_ptr<PyNode<State, Action, Reward>>> children;

    PyNode(const State                                        &state,
           const std::optional<Action>                        &action,
           const std::weak_ptr<PyNode<State, Action, Reward>> &parent,
           const RunningStats<Reward>                         &reward_stats)
        : NodeBase<PyNode, State, Action, Reward>(state, action, nullptr, reward_stats),
          parent(parent) {}

    explicit PyNode(const Node<State, Action, Reward> &other)
        : NodeBase<PyNode, State, Action, Reward>(other) {
        for (const auto &child: other.children) {
            this->children.push_back(std::make_shared<PyNode>(*child));
        }
    }

    PyNode(const PyNode &other) : NodeBase<PyNode, State, Action, Reward>(other) {
        for (const auto &child: other.children) {
            this->children.push_back(std::make_shared<PyNode>(*child));
        }
    }


    [[nodiscard]] size_t children_size() const { return children.size(); }

    explicit operator std::unique_ptr<Node<State, Action, Reward>>() const {
        return convert_py_node_to_node(*this);
    }

    // Serialization function definition
    template<class Archive>
    void serialize_derived(Archive &ar) {
        ar(cereal::make_nvp("children", children),
           cereal::make_nvp("parent", parent));
    }

    template<class Archive>
    static void
    load_and_construct_derived(Archive                                          &ar,
                               cereal::construct<PyNode<State, Action, Reward>> &construct,
                               State                                            &state,
                               std::optional<Action>                            &action,
                               RunningStats<Reward>                             &reward_stats

    ) {
        std::vector<std::shared_ptr<PyNode<State, Action, Reward>>> children;
        std::weak_ptr<PyNode<State, Action, Reward>>                parent;

        // Call deserialize on parent and children
        ar(cereal::make_nvp("children", children),
           cereal::make_nvp("parent", parent));
        construct(state, action, parent, reward_stats);
    }
};


template<typename State, typename Action, typename Reward>
std::unique_ptr<Node<State, Action, Reward>>
convert_py_node_to_node(const PyNode<State, Action, Reward> &py_node,
                        Node<State, Action, Reward>         *parent) {
    auto node          = std::make_unique<Node<State, Action, Reward>>(py_node.state,
                                                              py_node.action,
                                                              parent,
                                                              py_node.reward_stats);
    node->visit_count  = py_node.visit_count;
    node->update_count = py_node.update_count;
    node->is_terminal  = py_node.is_terminal;
    node->is_leaf      = py_node.is_leaf;
    for (const auto &py_child: py_node.children) {
        auto child = convert_py_node_to_node(*py_child, node.get());
        node->children.push_back(std::move(child));
    }
    return node;
}

template<typename State, typename Action, typename Reward>
std::shared_ptr<PyNode<State, Action, Reward>>
convert_node_to_py_node(const Node<State, Action, Reward>             &node,
                        std::shared_ptr<PyNode<State, Action, Reward>> parent) {
    auto py_node          = std::make_shared<PyNode<State, Action, Reward>>(node.state,
                                                                   node.action,
                                                                   parent,
                                                                   node.reward_stats);
    py_node->visit_count  = node.visit_count;
    py_node->update_count = node.update_count;
    py_node->is_terminal  = node.is_terminal;
    py_node->is_leaf      = node.is_leaf;
    for (const auto &child: node.children) {
        auto py_child = convert_node_to_py_node(*child, py_node);
        py_node->children.push_back(py_child);
    }
    return py_node;
}


#endif// DETAILS_MCTS_HPP