"""
Symbolic, auto differentiated implementation of the STTS in [1]

References:
    [1] https://arc.aiaa.org/doi/full/10.2514/1.G003897

"""

import itertools

import numpy as np
from sympy import Matrix, lambdify, diff, exp, symbols
from typing import List, Tuple, Dict, Callable, Union

# Cache to store lambdas
lambda_cache = {}


def get_or_create_lambda(eom: List, wrt_tuple: Tuple, vars: List) -> Callable:
    """
    Get or create lambda functions for given equations of motion (eom) and variables.

    Parameters
    ----------
    eom : list
        List of equations of motion.
    wrt_tuple : tuple
        Tuple of variables with respect to which differentiation will be done.
    wrt : list
        List of all variables.

    Returns
    -------
    function
        Lambda function corresponding to the differentiated equation.
    """

    key = (tuple(eom), wrt_tuple)
    if key not in lambda_cache:
        diff_expression = eom
        for var in wrt_tuple:
            diff_expression = diff(diff_expression, var)
        f_lambda = lambdify(vars, diff_expression, modules="numpy")
        lambda_cache[key] = f_lambda

    return lambda_cache[key]


def compute_tensors(
        eom: List,
        wrt: List,
        at: Dict[str, Union[float, np.ndarray]],
        vars: List = None,
        order: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute tensors up to a given order for the provided equations of motion (eom).

    Parameters
    ----------
    eom : list
        List of equations of motion.
    wrt : list
        List of variables with respect to which the tensor will be calculated.
    at : dict
        Dictionary containing the points at which to evaluate the tensors.
    order : int
        The highest order tensor to compute.

    Returns
    -------
    tuple
        Tensors up to the specified order.
    """

    p, q = len(eom), len(wrt)
    at_str = {str(k): v for k, v in at.items()}

    first_order = None
    second_order = None
    third_order = None
    fourth_order = None
    # print(at_str)
    # First order
    if order >= 1:
        first_order = np.zeros((p, q))
        for j in range(q):
            f_lambda = get_or_create_lambda(eom, (wrt[j],), vars)
            # print(f_lambda(**at_str).flatten())
            first_order[:, j] = f_lambda(**at_str).flatten()

    # Second order
    if order >= 2:
        second_order = np.zeros((p, q, q))
        for j in range(q):
            for k in range(j + 1):
                f_lambda = get_or_create_lambda(eom, (wrt[j], wrt[k]), vars)
                second_order[:, k, j] = second_order[:, j, k] = f_lambda(
                    **at_str
                ).flatten()

    # Third order
    if order >= 3:
        third_order = np.zeros((p, q, q, q))
        for j in range(q):
            for k in range(j + 1):
                for l in range(k + 1):
                    f_lambda = get_or_create_lambda(eom, (wrt[j], wrt[k], wrt[l]), vars)
                    base_value = f_lambda(**at_str).flatten()
                    for perm in itertools.permutations([j, k, l]):
                        third_order[:, perm[0], perm[1], perm[2]] = base_value

    # Fourth order
    if order >= 4:
        fourth_order = np.zeros((p, q, q, q, q))
        for j in range(q):
            for k in range(j + 1):
                for l in range(k + 1):
                    for m in range(l + 1):
                        f_lambda = get_or_create_lambda(
                            eom, (wrt[j], wrt[k], wrt[l], wrt[m]), vars
                        )
                        base_value = f_lambda(**at_str).flatten()
                        for perm in itertools.permutations([j, k, l, m]):
                            fourth_order[
                            :, perm[0], perm[1], perm[2], perm[3]
                            ] = base_value

    return first_order, second_order, third_order, fourth_order


from sympy import symbols, Matrix, exp

# Declare symbolic variables for position, velocity, and constants
x, y, z, vx, vy, vz = symbols("x y z vx vy vz", real=True)
GM_Earth, GM_Moon, CD, S, m, den, rho0, R_Earth, r0, J2 = symbols("GM_Earth GM_Moon CD S m den rho0 R_Earth r0 J2")

# Symbolic position and velocity vectors
r = Matrix([x, y, z])
v = Matrix([vx, vy, vz])

# Density function (symbolic)
rho_sym = rho0 * exp(-(r.norm() - r0) / den)

# Relative positions (symbolic)
sym_rel_moon = r - Matrix([384400e3, 0, 0])
sym_rel_earth = r - Matrix([0, 0, 0])

# Gravitational accelerations (symbolic)
sym_accel_earth = -GM_Earth * sym_rel_earth / (sym_rel_earth.norm() ** 3)
sym_accel_moon = -GM_Moon * sym_rel_moon / (sym_rel_moon.norm() ** 3)

# Drag acceleration (symbolic)
sym_drag = -0.5 * rho_sym * v.norm() * CD * S / m * v

# J2 Perturbation (symbolic)
r_norm = r.norm()
factor = 1.5 * J2 * (R_Earth / r_norm) ** 2
sym_j2_accel = factor * Matrix([
    (5 * (z / r_norm) ** 2 - 1) * x,
    (5 * (z / r_norm) ** 2 - 1) * y,
    (5 * (z / r_norm) ** 2 - 3) * z
]) * GM_Earth / (r_norm ** 3)

# Total acceleration (symbolic)
sym_accel = sym_accel_earth + sym_accel_moon + sym_j2_accel + sym_drag

# Final equations of motion (symbolic)
sym_eom = Matrix([vx, vy, vz, sym_accel[0], sym_accel[1], sym_accel[2]])


def compute_fourth_order_terms(f_i_j, f_i_jk, f_i_jkl, f_i_jklm, x_i_j, x_i_jk, x_i_jkl, x_i_jklm):
    # Rk4 for 4th-order tensor
    first_term_4th = np.einsum(
        "irspq,rj,sk,pl,qm->ijklm", f_i_jklm, x_i_j, x_i_j, x_i_j, x_i_j
    )
    second_term_part1_4th = np.einsum(
        "irsp,rjm,sk,pl->ijklm", f_i_jkl, x_i_jk, x_i_j, x_i_j
    )
    second_term_part2_4th = np.einsum(
        "irsp,rj,skm,sl->ijklm", f_i_jkl, x_i_j, x_i_jk, x_i_j
    )
    second_term_part3_4th = np.einsum(
        "irsp,rj,sk,plm->ijklm", f_i_jkl, x_i_j, x_i_j, x_i_jk
    )
    third_term_part1_4th = np.einsum("irsp,rjl,sk,pm->ijklm", f_i_jkl, x_i_jk, x_i_j, x_i_j)
    third_term_part2_4th = np.einsum("irsp,rj,skl,pm->ijklm", f_i_jkl, x_i_j, x_i_jk, x_i_j)
    third_term_part3_4th = np.einsum("irsp,rjk,sl,pm->ijklm", f_i_jkl, x_i_jk, x_i_j, x_i_j)
    fourth_term_part1_4th = np.einsum("irs,rjlm,sk->ijklm", f_i_jk, x_i_jkl, x_i_j)
    fourth_term_part2_4th = np.einsum("irs,rjl,skm->ijklm", f_i_jk, x_i_jk, x_i_jk)
    fourth_term_part3_4th = np.einsum("irs,rjm,skl->ijklm", f_i_jk, x_i_jk, x_i_jk)
    fourth_term_part4_4th = np.einsum("irs,rj,sklm->ijklm", f_i_jk, x_i_j, x_i_jkl)
    fourth_term_part5_4th = np.einsum("irs,rjkm,sl->ijklm", f_i_jk, x_i_jkl, x_i_j)
    fourth_term_part6_4th = np.einsum("irs,rjk,slm->ijklm", f_i_jk, x_i_jk, x_i_jk)
    fourth_term_part7_4th = np.einsum("irs,rjkl,sm->ijklm", f_i_jk, x_i_jkl, x_i_j)
    fifth_term_4th = np.einsum("ir,rjklm->ijklm", f_i_j, x_i_jklm)

    return (
            first_term_4th
            + second_term_part1_4th
            + second_term_part2_4th
            + second_term_part3_4th
            + third_term_part1_4th
            + third_term_part2_4th
            + third_term_part3_4th
            + fourth_term_part1_4th
            + fourth_term_part2_4th
            + fourth_term_part3_4th
            + fourth_term_part4_4th
            + fourth_term_part5_4th
            + fourth_term_part6_4th
            + fourth_term_part7_4th
            + fifth_term_4th
    )


def compute_third_order_terms(f_i_j, f_i_jk, f_i_jkl, x_i_j, x_i_jk, x_i_jkl):
    first_term = np.einsum("irsp,rj,sk,pl->ijkl", f_i_jkl, x_i_j, x_i_j, x_i_j)
    second_term_part1 = np.einsum("irs,rjl,sk->ijkl", f_i_jk, x_i_jk, x_i_j)
    second_term_part2 = np.einsum("irs,rj,skl->ijkl", f_i_jk, x_i_j, x_i_jk)
    second_term_part3 = np.einsum("irs,rjk,sl->ijkl", f_i_jk, x_i_jk, x_i_j)
    second_term = second_term_part1 + second_term_part2 + second_term_part3
    third_term = np.einsum("ir,rjkl->ijkl", f_i_j, x_i_jkl)
    xdot_i_jkl = first_term + second_term + third_term
    return xdot_i_jkl


def compute_second_order_terms(f_i_j, f_i_jk, x_i_j, x_i_jk):
    return np.einsum("irs,rj,sk->ijk", f_i_jk, x_i_j, x_i_j) + np.einsum("ir,rjk->ijk", f_i_j, x_i_jk)


def rk4_step(
        func, sym_func, state, x_i_j, x_i_jk, x_i_jkl, x_i_jklm, dt, at, wrt, vars, order=3
):
    """Runge-Kutta 4th order method with tensor propagation."""
    next_x_i_j = None
    next_x_i_jk = None
    next_x_i_jkl = None
    next_x_i_jklm = None

    # Pre-allocate memory for xdot terms
    if order >= 1:
        xdot_i_j = np.zeros_like(x_i_j)

    if order >= 2:
        xdot_i_jk = np.zeros_like(x_i_jk)

    if order >= 3:
        xdot_i_jkl = np.zeros_like(x_i_jkl)

    if order >= 4:
        xdot_i_jklm = np.zeros_like(x_i_jklm)

    def get_at(_state):
        return {
            **at,
            **{
                x: _state[0],
                y: _state[1],
                z: _state[2],
                vx: _state[3],
                vy: _state[4],
                vz: _state[5],
            },
        }

    # Standard RK4 for state
    k1 = func(*state).flatten()
    k2 = func(*state + k1 / 2 * dt).flatten()
    k3 = func(*state + k2 / 2 * dt).flatten()
    k4 = func(*state + k3 * dt).flatten()
    next_x = state + ((k1 + 2 * k2 + 2 * k3 + k4) / 6) * dt

    # Compute the Jacobian tensors at the RK4 stages
    f_i_j_1, f_i_jk_1, f_i_jkl_1, f_i_jklm_1 = compute_tensors(
        sym_func, wrt, get_at(state), vars, order
    )
    f_i_j_2, f_i_jk_2, f_i_jkl_2, f_i_jklm_2 = compute_tensors(
        sym_func, wrt, get_at(state + k1 / 2 * dt), vars, order
    )
    f_i_j_3, f_i_jk_3, f_i_jkl_3, f_i_jklm_3 = compute_tensors(
        sym_func, wrt, get_at(state + k2 / 2 * dt), vars, order
    )
    f_i_j_4, f_i_jk_4, f_i_jkl_4, f_i_jklm_4 = compute_tensors(
        sym_func, wrt, get_at(state + k3 * dt), vars, order
    )

    if order >= 1:
        # RK4 for 1st-order tensor
        # First stage
        x_i_j_1 = x_i_j
        xdot_i_j_1 = np.einsum("ij,jk->ik", f_i_j_1, x_i_j_1)

        # Second stage
        x_i_j_2 = x_i_j + xdot_i_j_1 * dt / 2
        xdot_i_j_2 = np.einsum("ij,jk->ik", f_i_j_2, x_i_j_2)

        # Third stage
        x_i_j_3 = x_i_j + xdot_i_j_2 * dt / 2
        xdot_i_j_3 = np.einsum("ij,jk->ik", f_i_j_3, x_i_j_3)

        # Fourth stage
        x_i_j_4 = x_i_j + xdot_i_j_3 * dt
        xdot_i_j_4 = np.einsum("ij,jk->ik", f_i_j_4, x_i_j_4)

        # Combine
        next_x_i_j = x_i_j + ((xdot_i_j_1 + 2 * xdot_i_j_2 + 2 * xdot_i_j_3 + xdot_i_j_4) / 6) * dt

    if order >= 2:
        # RK4 for 2nd-order tensor
        # First stage
        x_i_jk_1 = x_i_jk
        xdot_i_jk_1 = compute_second_order_terms(f_i_j_1, f_i_jk_1, x_i_j_1, x_i_jk_1)

        # Second stage
        x_i_jk_2 = x_i_jk + xdot_i_jk_1 * dt / 2
        xdot_i_jk_2 = compute_second_order_terms(f_i_j_2, f_i_jk_2, x_i_j_2, x_i_jk_2)

        # Third stage
        x_i_jk_3 = x_i_jk + xdot_i_jk_2 * dt / 2
        xdot_i_jk_3 = compute_second_order_terms(f_i_j_3, f_i_jk_3, x_i_j_3, x_i_jk_3)

        # Fourth stage
        x_i_jk_4 = x_i_jk + xdot_i_jk_3 * dt
        xdot_i_jk_4 = compute_second_order_terms(f_i_j_4, f_i_jk_4, x_i_j_4, x_i_jk_4)

        # Combine
        next_x_i_jk = x_i_jk + ((xdot_i_jk_1 + 2 * xdot_i_jk_2 + 2 * xdot_i_jk_3 + xdot_i_jk_4) / 6) * dt

    if order >= 3:
        # RK4 for 3rd-order tensor
        # First stage
        x_i_jkl_1 = x_i_jkl
        xdot_i_jkl_1 = compute_third_order_terms(f_i_j_1, f_i_jk_1, f_i_jkl_1, x_i_j_1, x_i_jk_1, x_i_jkl_1)

        # Second stage
        x_i_jkl_2 = x_i_jkl + xdot_i_jkl_1 * dt / 2
        xdot_i_jkl_2 = compute_third_order_terms(f_i_j_2, f_i_jk_2, f_i_jkl_2, x_i_j_2, x_i_jk_2, x_i_jkl_2)

        # Third stage
        x_i_jkl_3 = x_i_jkl + xdot_i_jkl_2 * dt / 2
        xdot_i_jkl_3 = compute_third_order_terms(f_i_j_3, f_i_jk_3, f_i_jkl_3, x_i_j_3, x_i_jk_3, x_i_jkl_3)

        # Fourth stage
        x_i_jkl_4 = x_i_jkl + xdot_i_jkl_3 * dt
        xdot_i_jkl_4 = compute_third_order_terms(f_i_j_4, f_i_jk_4, f_i_jkl_4, x_i_j_4, x_i_jk_4, x_i_jkl_4)

        # Combine all stages
        next_x_i_jkl = x_i_jkl + ((xdot_i_jkl_1 + 2 * xdot_i_jkl_2 + 2 * xdot_i_jkl_3 + xdot_i_jkl_4) / 6) * dt

    if order >= 4:
        # Rk4 for 4th-order tensor
        # First stage
        x_i_jklm_1 = x_i_jklm
        xdot_i_jklm_1 = compute_fourth_order_terms(f_i_j_1, f_i_jk_1, f_i_jkl_1, f_i_jklm_1, x_i_j_1, x_i_jk_1,
                                                   x_i_jkl_1, x_i_jklm_1)

        # Second stage
        x_i_jklm_2 = x_i_jklm + xdot_i_jklm_1 * dt / 2
        xdot_i_jklm_2 = compute_fourth_order_terms(f_i_j_2, f_i_jk_2, f_i_jkl_2, f_i_jklm_2, x_i_j_2, x_i_jk_2,
                                                   x_i_jkl_2, x_i_jklm_2)

        # Third stage
        x_i_jklm_3 = x_i_jklm + xdot_i_jklm_2 * dt / 2
        xdot_i_jklm_3 = compute_fourth_order_terms(f_i_j_3, f_i_jk_3, f_i_jkl_3, f_i_jklm_3, x_i_j_3, x_i_jk_3,
                                                   x_i_jkl_3, x_i_jklm_3)

        # Fourth stage
        x_i_jklm_4 = x_i_jklm + xdot_i_jklm_3 * dt
        xdot_i_jklm_4 = compute_fourth_order_terms(f_i_j_4, f_i_jk_4, f_i_jkl_4, f_i_jklm_4, x_i_j_4, x_i_jk_4,
                                                   x_i_jkl_4, x_i_jklm_4)

        # Combine all stages
        next_x_i_jklm = x_i_jklm + ((xdot_i_jklm_1 + 2 * xdot_i_jklm_2 + 2 * xdot_i_jklm_3 + xdot_i_jklm_4) / 6) * dt

    return next_x, next_x_i_j, next_x_i_jk, next_x_i_jkl, next_x_i_jklm


# Variables with respect to which differentiation may be needed
wrt = [x, y, z, vx, vy, vz]
vars = [x, y, z, vx, vy, vz]

# Constants as a dictionary to replace later
const_values = {
    GM_Earth: 3.986004418e14,
    GM_Moon: 4.9048695e12,
    CD: 2.0,
    S: 2000.0,
    m: 5000.0,
    den: 100000,
    rho0: 0,
    R_Earth: 6378.137e3,
    r0: 6771e3,
    J2: 1.08263e-3
}

# Substitute the constants when you need numerical evaluation
# sym_eom.subs(const_values)
sym_eom = sym_eom.subs(const_values)
eom = lambdify(vars, sym_eom, modules="numpy")

H = 300e3
r_p = 6371e3 + H
r0 = np.array([r_p, 0, 0])
# v_circle = np.sqrt(GM_Earth / np.linalg.norm(r0))
ecc = 0.02
sma = np.linalg.norm(r_p) / (1 - ecc)
# r_p = sma * (1 - ecc)
# print(r_p)
# v_a = np.sqrt(GM_Earth * (2 / np.linalg.norm(r_a) - 1 / sma))
v_p = np.sqrt(const_values[GM_Earth] * (2 / np.linalg.norm(r_p) - 1 / sma))
v0 = np.array([0, 0, 1]) * v_p

state0 = np.concatenate((r0, v0))

sst1_j0 = np.zeros((len(vars), len(wrt)))
np.fill_diagonal(sst1_j0, 1)
sst1_k0 = np.zeros((len(vars), len(wrt)))
sst2_jk0 = np.zeros((len(vars), len(wrt), len(wrt)))
sst3_jkl0 = np.zeros((len(vars), len(wrt), len(wrt), len(wrt)))
sst4_jklm0 = np.zeros((len(vars), len(wrt), len(wrt), len(wrt), len(wrt)))
up_to_order = 3

# Initialize
sst1_j = sst1_j0
sst2_jk = sst2_jk0
sst3_jkl = sst3_jkl0
sst4_jklm = sst4_jklm0

t_orbit = 2 * np.pi * np.sqrt(sma ** 3 / const_values[GM_Earth])
t = 0
dt = 100
trajectory = []
relevant_keys = ["x", "y", "z", "vx", "vy", "vz"]
# t_end = t_orbit * 2.5

jd = 86400
t_end = t_orbit * 1.0
# t_end = jd * 10

import time

start = time.time()
state = state0
while t < t_end:
    at = {
        x: state[0],
        y: state[1],
        z: state[2],
        vx: state[3],
        vy: state[4],
        vz: state[5],
    }

    # Update the next_state, next_sst1_j, next_sst2_jk
    state, sst1_j, sst2_jk, sst3_jkl, sst4_jklm = rk4_step(
        eom, sym_eom, state, sst1_j, sst2_jk, sst3_jkl, sst4_jklm, dt, at, wrt, vars, order=up_to_order
    )

    # Record the state
    trajectory.append(state)

    # Update the time
    t += dt

end = time.time()
print(f"Time elapsed: {end - start} seconds")
# Compute the terms using tensordot
# print("sst1_j:", sst1_j)
# print("sst2_jk:", sst2_jk)


# Simulate some data: Replace these with your actual calculations
# t_orbit, dt, state, GM_Earth, eom, dfdx, d2fdx2, wrt, sst1_j, sst2_jk would be defined here

# Initialize perturbations
# n_perturbations = 500
n_s = 2j
v_lim = 0
r_lim = 10e3
# n_perturbations = int(n_s.imag ** 6)
n_perturbations = 200

# # Grid sampling
# SX, SY, SZ, SVX, SVY, SVZ = np.mgrid[
#                             -v_lim:v_lim:n_s,
#                             -v_lim:v_lim:n_s,
#                             -v_lim:v_lim:n_s,
#                             -r_lim:r_lim:n_s,
#                             -r_lim:r_lim:n_s,
#                             -r_lim:r_lim:n_s]

dx0 = np.zeros((6, n_perturbations))

# multivariate normal distribution
sigma_pos = r_lim
sigma_vel = v_lim

# set seed
np.random.seed(0)
dx0 = np.random.multivariate_normal(
    np.zeros(6),
    np.diag([sigma_pos, sigma_pos, sigma_pos, sigma_vel, sigma_vel, sigma_vel]) ** 2,
    n_perturbations,
).T

print(dx0.shape)
# print(dx0)

# First order perturbations
dx_1 = np.einsum("ij,js->is", sst1_j, dx0)

# Second order; dimensions:
#  sst2_jk: (6, 6, 6)             (i, j, k)
#  1st dx0: (6, n_perturbations)  (j, s)
#  2nd dx0: (6, n_perturbations)  (k, s)
#  dx_1:    (6, n_perturbations)  (i, s)
dx_2 = np.einsum("ijk,js,ks->is", sst2_jk, dx0, dx0) / 2.0 + dx_1

# Third order
dx_3 = np.einsum("ijkl,js,ks,ls->is", sst3_jkl, dx0, dx0, dx0) / 6.0 + dx_2

# Forth order
if up_to_order >= 4:
    dx_4 = np.einsum("ijklm,js,ks,ls,ms->is", sst4_jklm, dx0, dx0, dx0, dx0) / 24.0 + dx_3
else:
    dx_4 = np.zeros((6, n_perturbations))

# List to hold all perturbed trajectories
perturbed_trajectories = []

# Iterate through each perturbation
for pert_idx in range(n_perturbations):
    perturbed_state = np.array(state0) + dx0[:, pert_idx]
    print(dx0[:, pert_idx])
    perturbed_trajectory = [perturbed_state]

    # Initialize the time and state for the perturbed simulation
    t_pert = 0.0

    while t_pert < t_end:
        at_pert = {
            x: perturbed_state[0],
            y: perturbed_state[1],
            z: perturbed_state[2],
            vx: perturbed_state[3],
            vy: perturbed_state[4],
            vz: perturbed_state[5],
        }

        # Update the next_state, for the perturbed trajectory
        perturbed_state, _, _, _, _ = rk4_step(
            eom,
            sym_eom,
            perturbed_state,
            sst1_j,
            sst2_jk,
            sst3_jkl,
            sst4_jklm,
            dt,
            at_pert,
            wrt,
            vars,
            order=0,
        )

        # Record the perturbed state
        perturbed_trajectory.append(perturbed_state)

        # Update the time
        t_pert += dt

    perturbed_trajectories.append(perturbed_trajectory)

final_perturbed_states = np.array(
    [trajectory[-1] for trajectory in perturbed_trajectories]
)

trajectory = np.array(trajectory)

# Assume trajectory is (n, m), n time steps and m dimensions
n, m = trajectory.shape

# # Original time steps
# t_old = np.linspace(0, 1, n)
#
# # New time steps (e.g., 300 points for a smoother curve)
# t_new = np.linspace(0, 1, 300)
#
# # Initialize smoothed_trajectory
# smoothed_trajectory = np.zeros((len(t_new), m))
#
# # Loop over each dimension
# for i in range(m):
#     # 1D data for the i-th dimension
#     y_old = trajectory[:, i]
#
#     # Generate cubic spline object
#     spline = make_interp_spline(t_old, y_old, k=3)
#
#     # Generate new y-values based on the new time steps
#     y_new = spline(t_new)
#
#     # Store this dimension's smoothed trajectory
#     smoothed_trajectory[:, i] = y_new
#
#
# trajectory = smoothed_trajectory


# Position Perturbations
axis_1 = 0
axis_2 = 2

axis_1_name = relevant_keys[axis_1]
axis_2_name = relevant_keys[axis_2]

numerical_dx = final_perturbed_states - trajectory[-1, :]

plot_with = 'matplotlib'

if plot_with == 'matplotlib':
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    # Define conversion factor
    meters_to_km = 1e-3

    matplotlib.use("QtAgg")
    nrows = 3
    ncols = 4 if up_to_order == 4 else 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 18), constrained_layout=True)

    marker_config = {
        "Initial Perturbations": {"c": "black", "s": 10, "alpha": 0.5},
        "SST1 Transformed": {"c": "red", "marker": "x", "s": 10, "alpha": 0.5},
        "SST2 Transformed": {"c": "blue", "s": 4, "alpha": 0.5},
        "SST3 Transformed": {"c": "green", "s": 10, "marker": "+", "alpha": 0.6},
        "SST4 Transformed": {"c": "orange", "s": 10, "marker": "o", "alpha": 0.5},
        "Numerically Perturbed": {"c": "purple", "marker": "s", "s": 10, "alpha": 0.5},
    }

    data_dicts = [
        {"dx": dx0, "name": "Initial Perturbations"},
        {"dx": dx_1, "name": "SST1 Transformed"},
        {"dx": dx_2, "name": "SST2 Transformed"},
        {"dx": dx_3, "name": "SST3 Transformed"},
    ]

    if up_to_order >= 4:
        data_dicts.append({"dx": dx_4, "name": "SST4 Transformed"})

    data_dicts.append({"dx": numerical_dx.T, "name": "Numerically Perturbed"})

    for ax_row in axes:
        for ax in ax_row:
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Light grid

    for i, ax_title, xlabel, ylabel in zip(
            range(ncols),
            ["Position Perturbations (km)", "Velocity Perturbations (km/s)", "Full Trajectory (km)"],
            [f"{axis_1_name.upper()} Position (km)", f"{axis_1_name.upper()} Velocity (km/s)",
             f"{axis_1_name.upper()} Position (km)"],
            [f"{axis_2_name.upper()} Position (km)", f"{axis_2_name.upper()} Velocity (km/s)",
             f"{axis_2_name.upper()} Position (km)"]
    ):
        ax = axes[0, i]  # First row
        if i == 2:
            ax.plot(trajectory[:, axis_1] * meters_to_km, trajectory[:, axis_2] * meters_to_km, c='black',
                    label='Nominal Trajectory')
        for data_dict in data_dicts:
            dx = data_dict['dx'] * meters_to_km  # Conversion to km or km/s
            name = data_dict['name']
            config = marker_config[name]

            axis_offset = 0
            if "Velocity" in ax_title:
                axis_offset = 3  # For velocity subplot

            x_data = dx[axis_1 + axis_offset, :]
            y_data = dx[axis_2 + axis_offset, :]

            if ax_title == "Full Trajectory (km)":  # For the full trajectory subplot
                x_data += trajectory[-1, axis_1] * meters_to_km
                y_data += trajectory[-1, axis_2] * meters_to_km

            ax.scatter(x_data, y_data, **config, label=name)

        # ax.axis("equal")
        ax.set_xlabel(xlabel, labelpad=15)
        ax.set_ylabel(ylabel, labelpad=15)
        ax.legend()
        ax.set_title(ax_title, pad=20)

    # SST-specific error plots
    for j, error_title in zip(range(1, 3), ["Position Errors (km)", "Velocity Errors (km/s)"]):
        for i, name in zip(range(1, ncols + 1),
                           ["SST1 Transformed", "SST2 Transformed", "SST3 Transformed", "SST4 Transformed"]):
            ax = axes[j, i - 1]
            axis_offset = 0
            if "Velocity" in error_title:
                axis_offset = 3  # For velocity subplot

            dx = data_dicts[i]['dx'] * meters_to_km  # Conversion to km or km/s
            numerical = data_dicts[-1]['dx'] * meters_to_km  # Conversion to km or km/s
            error_x = dx[axis_1 + axis_offset, :] - numerical[axis_1 + axis_offset, :]
            error_y = dx[axis_2 + axis_offset, :] - numerical[axis_2 + axis_offset, :]

            ax.scatter(error_x, error_y, **marker_config[name], label=f"{name.split(' ')[0]} Error")
            ax.axis("equal")
            ax.set_xlabel(f"Error in {axis_1_name.upper()} (km)", labelpad=15)
            ax.set_ylabel(f"Error in {axis_2_name.upper()} (km)", labelpad=15)
            ax.legend()
            ax.set_title(error_title, pad=20)

    plt.show()

elif plot_with == "mayavi":

    from mayavi import mlab

    mlab.figure(size=(400, 300))
    # Clear the current Mayavi scene
    # mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    data_norm = np.linalg.norm(trajectory, axis=0)[None, ...]
    print(data_norm)
    data_norm = np.ones((1, 6)) * np.max(data_norm)
    # Create the 3D scatter plot for the position perturbations
    # mlab.points3d(dx0 / data_norm.T, color=(0, 0, 0), scale_factor=0.1, opacity=0.5)
    # mlab.points3d(dx_1, color=(1, 0, 0), scale_factor=0.1, opacity=0.5)
    # mlab.points3d(dx_2, color=(0, 0, 1), scale_factor=0.1, opacity=0.5)
    # mlab.points3d(dx_3, color=(0, 1, 0), scale_factor=0.1, opacity=0.5)

    # # Create the 3D scatter plot for the velocity perturbations
    # mlab.points3d(dx0[3 + axis_1, :], dx0[3 + axis_2, :], color=(0, 0, 0), scale_factor=0.1, opacity=0.5)
    # mlab.points3d(dx_1[3 + axis_1, :], dx_1[3 + axis_2, :], color=(1, 0, 0), scale_factor=0.1, opacity=0.5)
    # mlab.points3d(dx_2[3 + axis_1, :], dx_2[3 + axis_2, :], color=(0, 0, 1), scale_factor=0.1, opacity=0.5)
    # mlab.points3d(dx_3[3 + axis_1, :], dx_3[3 + axis_2, :], color=(0, 1, 0), scale_factor=0.1, opacity=0.5)
    #
    # x = trajectory[:, 0]
    # y = trajectory[:, 1]
    # z = trajectory[:, 2]
    # s = np.linspace(0.5, 1.0, len(trajectory))
    # # Create the 3D plot for the full trajectory with position perturbations
    # mlab.plot3d(x, y, z)
    # n_mer, n_long = 6, 11
    # dphi = np.pi / 1000.0
    # phi = np.arange(0.0, 2 * np.pi + 0.5 * dphi, dphi)
    # mu = phi * n_mer
    # x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    # y = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    # z = np.sin(n_long * mu / n_mer) * 0.5

    # l = mlab.plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap='Spectral')
    # plot test
    # mlab.points3d([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3])

    # every third point
    # normalize trajectory dimensions to [-1, 1]
    trajectory = np.array(trajectory)
    trajectory = trajectory / data_norm

    # trajectory = trajectory[::3, :]
    # print(len(trajectory))
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    s = np.linspace(0.5, 1.0, len(trajectory))
    # mlab.points3d(x, y, z, s, colormap="Spectral", scale_factor=0.1, tube_radius=0.025)
    dx0 /= data_norm.T
    dx_1 /= data_norm.T
    dx_1 += trajectory[-1, :][..., None]
    dx_2 /= data_norm.T
    dx_2 += trajectory[-1, :][..., None]
    dx_3 /= data_norm.T
    dx_3 += trajectory[-1, :][..., None]
    numerical_dx /= data_norm
    numerical_dx += trajectory[-1, :][None, ...]

    # mlab.points3d(*dx0[:3,:], color=(0, 0, 0), scale_factor=0.001, opacity=0.5)
    mlab.points3d(*dx_1[:3, :], color=(1, 0, 0), scale_factor=0.001, opacity=0.1)
    mlab.points3d(*dx_2[:3, :], color=(0, 0, 1), scale_factor=0.001, opacity=0.1)
    mlab.points3d(*dx_3[:3, :], color=(0, 1, 0), scale_factor=0.001, opacity=0.1)
    mlab.points3d(
        *numerical_dx[:, :3].T, color=(0.5, 0, 0.5), scale_factor=0.001, opacity=0.1
    )
    mlab.plot3d(x, y, z, s, colormap="Spectral", tube_radius=0.0005, opacity=0.5)

    # Add other features like legends, titles, etc., using mlab.text3d(), mlab.title(), etc.

    # Show the 3D plot
    # mlab.view(azimuth=0, elevation=90, distance=1000)
    mlab.show()
