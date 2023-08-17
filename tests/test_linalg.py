import numpy as np

import pydin.core.linalg as pdl


# Add the necessary fixtures for creating test data if needed.

def test_linspace():
    # Basic Test
    expected = np.linspace(1., 5., 5, True)
    np.testing.assert_allclose(pdl.linspace(1., 5., 5, True), expected)


def test_logspace():
    # Basic Test
    expected = np.logspace(1., 5., 5, True)
    np.testing.assert_allclose(pdl.logspace(1., 5., 5, True), expected)


def test_geomspace():
    # Basic Test
    expected = np.geomspace(1., 5., 5, True)
    np.testing.assert_allclose(pdl.geomspace(1., 5., 5, True), expected)


def test_meshgrid():
    # Basic Test
    x = np.linspace(1., 3., 3)
    y = np.linspace(4., 6., 3)
    expected_X, expected_Y = np.meshgrid(x, y, indexing='xy')
    X, Y = pdl.meshgrid(x, y, parallel=True)
    np.testing.assert_allclose(X, expected_X)
    np.testing.assert_allclose(Y, expected_Y)

    # Large Num Test
    x = np.linspace(0., 1e3, int(1e3 + 1))
    y = np.linspace(0., 1e3, int(1e3 + 1))
    expected_X, expected_Y = np.meshgrid(x, y, indexing='xy')
    X, Y = pdl.meshgrid(x, y, parallel=True)
    np.testing.assert_allclose(X, expected_X)
    np.testing.assert_allclose(Y, expected_Y)


#
#     # ...add more meshgrid tests as needed
#

if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
