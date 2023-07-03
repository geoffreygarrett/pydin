import numpy as np
import pydin

# Constants
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
TOL = 1e-5
mu = 398600.4418  # Earth's gravitational constant in km^3/s^2
e = 0.01  # A small eccentricity


def test_angle_conversions():
    # Original angles in radians
    mean_orig = 45.0 * DEG_TO_RAD  # Mean anomaly
    eccentric_orig = pydin.anomaly_mean_to_eccentric(mean_orig, e)  # Eccentric anomaly
    true_orig = pydin.anomaly_eccentric_to_true(eccentric_orig, e)  # True anomaly

    # Convert back to mean anomaly from eccentric anomaly
    mean_rt = pydin.anomaly_eccentric_to_mean(eccentric_orig, e)

    # Convert back to eccentric anomaly from mean anomaly
    eccentric_rt = pydin.anomaly_mean_to_eccentric(mean_rt, e)

    # Convert back to true anomaly from eccentric anomaly
    true_rt = pydin.anomaly_eccentric_to_true(eccentric_rt, e)

    # Convert back to eccentric anomaly from true anomaly
    eccentric_rt2 = pydin.anomaly_true_to_eccentric(true_rt, e)

    # Convert back to mean anomaly from true anomaly
    mean_rt2 = pydin.anomaly_true_to_mean(true_rt, e)

    # Convert back to true anomaly from mean anomaly
    true_rt2 = pydin.anomaly_mean_to_true(mean_rt2, e)

    # Check that the original and round-trip values match, within some tolerance
    assert abs(mean_orig - mean_rt) < TOL
    assert abs(eccentric_orig - eccentric_rt) < TOL
    assert abs(true_orig - true_rt) < TOL
    assert abs(eccentric_orig - eccentric_rt2) < TOL
    assert abs(mean_orig - mean_rt2) < TOL
    assert abs(true_orig - true_rt2) < TOL


def test_round_trip_case1():
    # Original position and velocity
    r_orig = np.array([6524.834, 6862.875, 6448.296])  # km
    v_orig = np.array([4.901327, 5.533756, -1.976341])  # km/s

    # Convert to classical orbital elements
    p, e, i, raan, argp, nu = pydin.rv2coe(mu, r_orig, v_orig)

    # Now convert back to position and velocity
    r, v = pydin.coe2rv(mu, p, e, i, raan, argp, nu)

    # Check that the original and round-trip values match, within some tolerance
    for j in range(3):
        assert abs(r[j] - r_orig[j]) < TOL
        assert abs(v[j] - v_orig[j]) < TOL

    # Convert position and velocity back to orbital elements
    p_rt, e_rt, i_rt, raan_rt, argp_rt, nu_rt = pydin.rv2coe(mu, r, v)

    # Check that the original and round-trip orbital elements match, within some tolerance
    assert abs(p - p_rt) < TOL
    assert abs(e - e_rt) < TOL
    assert abs(i - i_rt) < TOL
    assert abs(raan - raan_rt) < TOL
    assert abs(argp - argp_rt) < TOL
    assert abs(nu - nu_rt) < TOL


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
