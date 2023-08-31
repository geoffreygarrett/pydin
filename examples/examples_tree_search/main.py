import numpy as np
import numba as nb

# General Constants
MU = 1.32712440018e11  # Gravitational parameter of the sun, km^3/s^2
G_0 = 9.80665  # Standard gravitational acceleration, m/s^2
AU = 1.49597870691e8  # Astronomical unit, km
DAY_TO_SEC = 86400.0  # Day, seconds
YEAR_TO_DAY = 365.25  # Year, seconds

# Conversion factor from degrees to radians
DEG_TO_RAD = np.pi / 180.0

# t = 64328 MJD
# Planetary Constants for Venus, Earth, and Mars
PLANETS = {
    "Venus": {
        "mu": 3.24858592000e5,  # km^3/s^2
        "R_min": 6351.0,  # km
        "a": 1.08208010521e8,  # km
        "e": 6.72988099539e-3,
        "i": 3.39439096544 * DEG_TO_RAD,  # radians
        "RAAN": 7.65796397775e1 * DEG_TO_RAD,  # radians
        "arg_perihelion": 5.51107191497e1 * DEG_TO_RAD,  # radians
        "M": 1.11218416921e1 * DEG_TO_RAD,  # radians
    },
    "Earth": {
        "mu": 3.98600435436e5,  # km^3/s^2
        "R_min": 6678.0,  # km
        "a": 1.49579151285e8,  # km
        "e": 1.65519129162e-2,
        "i": 4.64389155500e-3 * DEG_TO_RAD,  # radians
        "RAAN": 1.98956406477e2 * DEG_TO_RAD,  # radians
        "arg_perihelion": 2.62960364700e2 * DEG_TO_RAD,  # radians
        "M": 3.58039899470e2 * DEG_TO_RAD,  # radians
    },
    "Mars": {
        "mu": 4.28283752140e4,  # km^3/s^2
        "R_min": 3689.0,  # km
        "a": 2.27951663551e8,  # km
        "e": 9.33662184095e-2,
        "i": 1.84693231241 * DEG_TO_RAD,  # radians
        "RAAN": 4.94553142513e1 * DEG_TO_RAD,  # radians
        "arg_perihelion": 2.86731029267e2 * DEG_TO_RAD,  # radians
        "M": 2.38232037154e2 * DEG_TO_RAD,  # radians
    },
}

# Constraint constants
I_SP = 4000.0  # s
T_MAX = 0.6  # N
RHO = 0.004  # kg-1
K = 10.0  # kg/yr
M_D = 500  # kg, constant dry mass
M_S = 40  # kg, mass of a miner
MASS_MAX = 3000  # kg, maximum mass allowed
TIME_MAX = 15 * YEAR_TO_DAY * DAY_TO_SEC  # s, maximum mission time
# TODO: Further constraint to be counted elsewhere. Asteroids can only be visited twice.

# Tolerances
POS_TOL = 1_000  # km
VEL_TOL = 1.0  # km/s
MASS_TOL = 1e-3  # kg


@nb.njit(nogil=True)
def max_ships_for_mass(avg_mass):
    return np.min(100, 2 * np.exp(avg_mass / RHO))


@nb.njit(nogil=True)
def total_resource_value(c_bonus, m_asteroids):
    return np.sum(c_bonus * m_asteroids)


@nb.njit(nogil=True)
def max_collected_mass(t1, t2):
    return K * (t2 - t1)


@nb.njit(nogil=True)
def m_0(m_p, n_miners):
    _m_0 = M_D + m_p + n_miners * M_S

    if _m_0 > MASS_MAX:
        # calculate how much propellant or miners should be decreased to be within the mass limit
        excess_mass = _m_0 - MASS_MAX
        decrease_propellant = excess_mass
        decrease_miners = np.ceil(excess_mass / M_S)

        return {
            'status': 'Overweight',
            'current_mass': _m_0,
            'decrease_propellant': decrease_propellant,
            'decrease_miners': decrease_miners
        }
    else:
        return {
            'status': 'OK',
            'current_mass': _m_0
        }


@nb.njit(nogil=True)
def termination_sun_distance(sun_distance):
    return sun_distance < 0.3 * AU


@nb.njit(nogil=True)
def termination_interval_rendeavous(t1, t2):
    return t2 - t1 < 1 * YEAR_TO_DAY * DAY_TO_SEC


