import numpy as np
import pytest

from benchmarks import SingleParameter, ArrayParameter, GridParameter


# testing SingleParameter
def test_single_param_negative():
    single = SingleParameter(-10)
    assert single.values() == [-10]


def test_single_param_str():
    single = SingleParameter('test')
    assert single.values() == ['test']


def test_single_param_empty():
    single = SingleParameter('')
    assert single.values() == ['']


def test_single_param_none():
    single = SingleParameter(None)
    assert single.values() == [None]


def test_single_param_boolean():
    single = SingleParameter(True)
    assert single.values() == [True]


# testing ArrayParameter
def test_array_param_negative():
    array = ArrayParameter(np.linspace(-1, 0, 5))
    assert array.values() == [-1.0, -0.75, -0.5, -0.25, 0.0]


def test_array_param_str():
    array = ArrayParameter(['a', 'b', 'c'])
    assert array.values() == ['a', 'b', 'c']


def test_array_param_mixed_types():
    array = ArrayParameter(['a', 1, None])
    assert array.values() == ['a', 1, None]


def test_array_param_empty():
    array = ArrayParameter([])
    assert array.values() == []


def test_array_param_boolean():
    array = ArrayParameter([True, False, True])
    assert array.values() == [True, False, True]


def test_array_param_none():
    array = ArrayParameter([None, None, None])
    assert array.values() == [None, None, None]


# testing GridParameter
def test_grid_param_mix():
    grid = GridParameter([SingleParameter(-10), ArrayParameter(np.linspace(0, 1, 5))])
    expected_grid_values = [
        [-10, -10, -10, -10, -10],
        [0.0, 0.25, 0.5, 0.75, 1.0]
    ]
    assert grid.values() == expected_grid_values


def test_grid_param_str():
    grid = GridParameter([SingleParameter('test'), ArrayParameter(['a', 'b', 'c'])])
    expected_grid_values = [
        ['test', 'test', 'test'],
        ['a', 'b', 'c']
    ]
    assert grid.values() == expected_grid_values


def test_grid_param_empty():
    grid = GridParameter([])
    assert grid.values() == []


def test_grid_param_none():
    grid = GridParameter([SingleParameter(None), ArrayParameter([None, None, None])])
    expected_grid_values = [
        [None, None, None],
        [None, None, None]
    ]
    assert grid.values() == expected_grid_values


def test_grid_param_iterator():
    single_param = SingleParameter(10)
    array_param = ArrayParameter(np.linspace(0, 1, 2))  # [0.0, 1.0]
    grid = GridParameter([single_param, array_param])

    expected_combinations = [
        (10, 0.0),
        (10, 1.0)
    ]

    for i, params in enumerate(grid):
        assert params == expected_combinations[i]


def test_array_param_numpy_array():
    array = ArrayParameter(np.array([1, 2, 3]))
    assert array.values() == [1, 2, 3]


def test_grid_param_single_element_array():
    array = ArrayParameter([1])
    grid = GridParameter([SingleParameter('test'), array])
    expected_grid_values = [
        ['test'],
        [1]
    ]
    assert grid.values() == expected_grid_values


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
