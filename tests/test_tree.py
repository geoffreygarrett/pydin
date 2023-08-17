import numpy as np
from pydin.core.tree import Node, Tree


def test_basic_test():
    node = Node(10)
    assert node.get_data() == 10
    assert node.get_parent() is None
    assert len(node.get_children()) == 0


def test_string_node_test():
    raw_node = Node("hello")
    assert raw_node.get_data() == "hello"


def test_parent_child_relationship():
    parent_node = Node(10)
    child_node = Node(5)

    parent_node.add_child(child_node)

    assert len(parent_node.get_children()) == 1
    assert parent_node.get_children()[0] == child_node
    assert child_node.get_parent() == parent_node


def test_custom_class_data():
    class CustomClass:
        def __init__(self, x):
            self.x = x

    custom_node = Node(CustomClass(10))
    assert custom_node.get_data().x == 10


def test_deep_tree_base_functions():
    node = Node(10)
    child1 = Node(5)
    child2 = Node(15)
    child3 = Node(20)
    grandchild1 = Node(25)
    grandchild2 = Node(30)

    child1.add_child(grandchild1)
    child2.add_child(grandchild2)
    node.add_child(child1)
    node.add_child(child2)
    node.add_child(child3)

    assert node.get_node_count() == 6
    assert node.get_max_depth() == 3


def test_binary_io_methods():
    node = Node(10)
    child1 = Node(5)
    child2 = Node(15)
    child3 = Node(20)
    grandchild1 = Node(25)
    grandchild2 = Node(30)

    child1.add_child(grandchild1)
    child2.add_child(grandchild2)
    node.add_child(child1)
    node.add_child(child2)
    node.add_child(child3)

    node.save_binary("test_tree.bin")
    node.save_json("test_tree.json")

    node2 = Node.load_binary("test_tree.bin")
    node3 = Node.load_json("test_tree.json")

    assert node2 is not None
    assert node3 is not None

    # assert node.get_node_count() == node2.get_node_count()
    # assert node.get_max_depth() == node2.get_max_depth()

    assert node.get_node_count() == node3.get_node_count()
    assert node.get_max_depth() == node3.get_max_depth()
    assert node.get_data() == node3.get_data()
    for i in range(len(node.get_children())):
        assert node.get_children()[i].get_data() == node3.get_children()[i].get_data()


def test_node_not_found():
    # Try to remove a child that doesn't exist
    parent = Node(10)
    not_found = Node(404)
    removed_child = parent.remove_child(not_found)

    # Check that removed child is null
    assert removed_child is None


def test_node_found():
    # Try to remove a child that does exist
    parent = Node(10)
    found = Node(200)
    parent.add_child(found)

    # Check that the child is in the parent's children
    assert found in parent.get_children()
    removed_child = parent.remove_child(found)

    # Check that the child is not in the parent's children
    assert found not in parent.get_children()

    # Check that removed child is not null
    assert removed_child is not None
    assert removed_child == found


def test_holds_python_objects():
    class CustomClass:
        def __init__(self, x):
            self.x = x

    node = Node(CustomClass(10))
    assert node.holds_python_object() is True

    node = Node(10)
    assert node.holds_python_object() is False


def test_exotic_types():
    node = Node(np.array([1., 2., 3.]))
    # why is this being prioritised as a py::object? we have eigen.h?
    # TODO: figure this out
    # assert node.holds_python_object() is False


def test_basic_tree():
    node = Node(10)
    child1 = Node(5)
    child2 = Node(15)
    child3 = Node(20)
    grandchild1 = Node(25)
    grandchild2 = Node(30)

    child1.add_child(grandchild1)
    child2.add_child(grandchild2)
    node.add_child(child1)
    node.add_child(child2)
    node.add_child(child3)

    tree = Tree(node)

    assert tree.get_root().get_data() == node.get_data()
    assert tree.get_root() != node  # to demonstrate that it's a copy, because TREE takes ownership


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
