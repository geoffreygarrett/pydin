import itertools

import numpy as np
from sympy import Matrix, lambdify, diff, exp, symbols

# Cache to store lambdas
lambda_cache = {}


# Helper function to get or generate lambda
def get_or_create_lambda(eom, wrt_tuple, wrt):
    key = (tuple(eom), tuple(wrt_tuple))

    if key not in lambda_cache:
        diff_expression = eom
        for var in wrt_tuple:
            diff_expression = diff(diff_expression, var)
        f_lambda = lambdify(wrt, diff_expression, modules="numpy")
        lambda_cache[key] = f_lambda
    return lambda_cache[key]


def compute_tensors(eom, wrt, at, order):
    p, q = len(eom), len(wrt)
    at_str = {str(k): v for k, v in at.items()}

    first_order = None
    second_order = None
    third_order = None
    fourth_order = None

    # First order
    if order >= 1:
        first_order = np.zeros((p, q))
        for j in range(q):
            f_lambda = get_or_create_lambda(eom, (wrt[j],), wrt)
            first_order[:, j] = f_lambda(**at_str).flatten()

    # Second order
    if order >= 2:
        second_order = np.zeros((p, q, q))
        for j in range(q):
            for k in range(j + 1):
                f_lambda = get_or_create_lambda(eom, (wrt[j], wrt[k]), wrt)
                second_order[:, k, j] = second_order[:, j, k] = f_lambda(
                    **at_str
                ).flatten()

    # Third order
    if order >= 3:
        third_order = np.zeros((p, q, q, q))
        for j in range(q):
            for k in range(j + 1):
                for l in range(k + 1):
                    f_lambda = get_or_create_lambda(eom, (wrt[j], wrt[k], wrt[l]), wrt)
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
                            eom, (wrt[j], wrt[k], wrt[l], wrt[m]), wrt
                        )
                        base_value = f_lambda(**at_str).flatten()
                        for perm in itertools.permutations([j, k, l, m]):
                            fourth_order[
                                :, perm[0], perm[1], perm[2], perm[3]
                            ] = base_value

    return first_order, second_order, third_order, fourth_order


derivative_cache = {}


def get_derivative(eom, wrt, order, at):
    cache_key = (tuple(eom), tuple(wrt), order, tuple(at.keys()))

    if cache_key in derivative_cache:
        return derivative_cache[cache_key]

    combinations = list(itertools.product(wrt, repeat=order))
    # combinations = [tuple(sorted(combination)) for combination in combinations]
    # combinations = unique_combinations(wrt, order)
    print(f"Computing {len(combinations)} derivatives of order {order}")
    print(combinations)
    derivatives = []

    for eq in eom:
        eq_derivs = [diff(eq, *combination) for combination in combinations]
        derivatives.append(eq_derivs)

    # at_str = {str(k): v for k, v in at.items()}
    f_lambdified = lambdify(wrt, derivatives, modules="numpy")
    derivative_cache[cache_key] = f_lambdified

    return f_lambdified


def unique_combinations(elements, repeat):
    unique_comb = set()
    for combination in itertools.product(elements, repeat=repeat):
        sorted_comb = tuple(sorted([*combination]))
        unique_comb.add(sorted_comb)
    return list(unique_comb)


GM_Earth = 3.986004418e14
GM_Moon = 4.9048695e12
CD = 2.0
S = 200.0
m = 5000.0
r_moon = np.array([384400e3, 0, 0])
r_earth = np.array([0, 0, 0])
J2 = 1.08263e-3  # Earth's second zonal harmonic
R_Earth = 6378.137e3  # Earth's radius in meters
den = 1000000
r0 = 6771e3
rho0 = 3.8 * 10**-12


def rho_func(r):
    # res = rho0
    res = rho0 * np.exp(-(np.linalg.norm(r) - r0) / den)
    return res


def eom(state: np.ndarray, dt: float):
    rel_moon = state[:3] - r_moon
    rel_earth = state[:3] - r_earth
    accel_moon = -GM_Moon * rel_moon / (np.linalg.norm(rel_moon) ** 3)
    accel_earth = -GM_Earth * rel_earth / (np.linalg.norm(rel_earth) ** 3)
    accel_drag = (
        -0.5 * rho_func(state[:3]) * np.linalg.norm(state[3:]) * CD * S / m * state[3:]
    )

    # J2 Perturbation for Earth
    x, y, z = state[:3]
    r_norm = np.linalg.norm(state[:3])
    factor = 1.5 * J2 * (R_Earth / r_norm) ** 2
    j2_term_x = factor * (5 * (z / r_norm) ** 2 - 1) * x
    j2_term_y = factor * (5 * (z / r_norm) ** 2 - 1) * y
    j2_term_z = factor * (5 * (z / r_norm) ** 2 - 3) * z
    j2_accel = np.array([j2_term_x, j2_term_y, j2_term_z]) * GM_Earth / (r_norm**3)

    accel_total = accel_moon + accel_earth + j2_accel + accel_drag
    return np.concatenate((state[3:], accel_total))


# Define physical parameters and symbols
x, y, z, vx, vy, vz = symbols("x y z vx vy vz", real=True)
r = Matrix([x, y, z])
v = Matrix([vx, vy, vz])
rho_sym = rho0 * exp(-(r.norm() - r0) / den)

# Symbolic equations for autodiff
sym_rel_moon = Matrix([x, y, z]) - Matrix(r_moon)
sym_rel_earth = Matrix([x, y, z]) - Matrix(r_earth)
sym_accel_earth = -GM_Earth * sym_rel_earth / (sym_rel_earth.norm() ** 3)
sym_accel_moon = -GM_Moon * sym_rel_moon / (sym_rel_moon.norm() ** 3)
sym_drag = -0.5 * rho_sym * v.norm() * CD * S / m * v
r_norm = r.norm()

# J2 Perturbation for Earth
factor = 1.5 * J2 * (R_Earth / r_norm) ** 2
sym_j2_term_x = factor * (5 * (z / r_norm) ** 2 - 1) * x
sym_j2_term_y = factor * (5 * (z / r_norm) ** 2 - 1) * y
sym_j2_term_z = factor * (5 * (z / r_norm) ** 2 - 3) * z
sym_j2_accel = (
    Matrix([sym_j2_term_x, sym_j2_term_y, sym_j2_term_z]) * GM_Earth / (r_norm**3)
)

# Previous accelerations
sym_accel = sym_accel_earth + sym_accel_moon + sym_j2_accel + sym_drag
sym_eom = Matrix([vx, vy, vz, sym_accel[0], sym_accel[1], sym_accel[2]])
wrt = [x, y, z, vx, vy, vz]



def compute_third_order_terms(f_i_j, f_i_jk, f_i_jkl, x_i_j, x_i_jk, x_i_jkl):
    first_term = np.einsum("irsp,rj,sk,pl->ijkl", f_i_jkl, x_i_j, x_i_j, x_i_j)
    second_term_part1 = np.einsum("irs,rjl,sk->ijkl", f_i_jk, x_i_jk, x_i_j)
    second_term_part2 = np.einsum("irs,rj,skl->ijkl", f_i_jk, x_i_j, x_i_jk)
    second_term_part3 = np.einsum("irs,rjk,sl->ijkl", f_i_jk, x_i_jk, x_i_j)
    second_term = second_term_part1 + second_term_part2 + second_term_part3
    third_term = np.einsum("ir,rjkl->ijkl", f_i_j, x_i_jkl)
    xdot_i_jkl = first_term + second_term + third_term
    return xdot_i_jkl

def rk4_step(
    func, sym_func, state, x_i_j, x_i_jk, x_i_jkl, x_i_jklm, dt, at, wrt, order=3
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
    k1 = func(state, dt)
    k2 = func(state + k1 / 2 * dt, dt)
    k3 = func(state + k2 / 2 * dt, dt)
    k4 = func(state + k3 * dt, dt)
    next_x = state + ((k1 + 2 * k2 + 2 * k3 + k4) / 6) * dt

    # Compute the Jacobian tensors at the RK4 stages
    f_i_j_1, f_i_jk_1, f_i_jkl_1, f_i_jklm_1 = compute_tensors(
        sym_func, wrt, get_at(state), order
    )
    f_i_j_2, f_i_jk_2, f_i_jkl_2, f_i_jklm_2 = compute_tensors(
        sym_func, wrt, get_at(state + k1 / 2 * dt), order
    )
    f_i_j_3, f_i_jk_3, f_i_jkl_3, f_i_jklm_3 = compute_tensors(
        sym_func, wrt, get_at(state + k2 / 2 * dt), order
    )
    f_i_j_4, f_i_jk_4, f_i_jkl_4, f_i_jklm_4 = compute_tensors(
        sym_func, wrt, get_at(state + k3 * dt), order
    )

    if order >= 1:
        # RK4 for 1st-order tensor
        xdot_i_j_1 = np.einsum("ij,jk->ik", f_i_j_1, x_i_j)
        xdot_i_j_2 = np.einsum("ij,jk->ik", f_i_j_2, x_i_j + xdot_i_j_1 * dt / 2)
        xdot_i_j_3 = np.einsum("ij,jk->ik", f_i_j_3, x_i_j + xdot_i_j_2 * dt / 2)
        xdot_i_j_4 = np.einsum("ij,jk->ik", f_i_j_4, x_i_j + xdot_i_j_3 * dt)
        next_x_i_j = x_i_j + ((xdot_i_j_1 + 2 * xdot_i_j_2 + 2 * xdot_i_j_3 + xdot_i_j_4) / 6) * dt

    if order >= 2:
        # RK4 for 2nd-order tensor
        xdot_i_jk_1 = np.einsum("irs,rj,sk->ijk", f_i_jk_1, x_i_j, x_i_j) + np.einsum("ir,rjk->ijk", f_i_j_1, x_i_jk)
        xdot_i_jk_2 = np.einsum("irs,rj,sk->ijk", f_i_jk_2, x_i_j + xdot_i_j_1 * dt / 2, x_i_j + xdot_i_j_1 * dt / 2) + np.einsum("ir,rjk->ijk", f_i_j_2, x_i_jk + xdot_i_jk_1 * dt / 2)
        xdot_i_jk_3 = np.einsum("irs,rj,sk->ijk", f_i_jk_3, x_i_j + xdot_i_j_2 * dt / 2, x_i_j + xdot_i_j_2 * dt / 2) + np.einsum("ir,rjk->ijk", f_i_j_3, x_i_jk + xdot_i_jk_2 * dt / 2)
        xdot_i_jk_4 = np.einsum("irs,rj,sk->ijk", f_i_jk_4, x_i_j + xdot_i_j_3 * dt, x_i_j + xdot_i_j_3 * dt) + np.einsum("ir,rjk->ijk", f_i_j_4, x_i_jk + xdot_i_jk_3 * dt)
        next_x_i_jk = x_i_jk + ((xdot_i_jk_1 + 2 * xdot_i_jk_2 + 2 * xdot_i_jk_3 + xdot_i_jk_4) / 6) * dt


    if order >= 3:
        # RK4 for 3rd-order tensor
        # First stage
        xdot_i_jkl_1 = compute_third_order_terms(f_i_j_1, f_i_jk_1, f_i_jkl_1, x_i_j, x_i_jk, x_i_jkl)

        # Second stage
        xdot_i_jkl_2 = compute_third_order_terms(f_i_j_2, f_i_jk_2, f_i_jkl_2, x_i_j + xdot_i_j_1 * dt / 2, x_i_jk + xdot_i_jk_1 * dt / 2, x_i_jkl + xdot_i_jkl_1 * dt / 2)

        # Third stage
        xdot_i_jkl_3 = compute_third_order_terms(f_i_j_3, f_i_jk_3, f_i_jkl_3, x_i_j + xdot_i_j_2 * dt / 2, x_i_jk + xdot_i_jk_2 * dt / 2, x_i_jkl + xdot_i_jkl_2 * dt / 2)

        # Fourth stage
        xdot_i_jkl_4 = compute_third_order_terms(f_i_j_4, f_i_jk_4, f_i_jkl_4, x_i_j + xdot_i_j_3 * dt, x_i_jk + xdot_i_jk_3 * dt, x_i_jkl + xdot_i_jkl_3 * dt)

        # Combine all stages
        next_x_i_jkl = x_i_jkl + ((xdot_i_jkl_1 + 2 * xdot_i_jkl_2 + 2 * xdot_i_jkl_3 + xdot_i_jkl_4) / 6) * dt


    if order >= 4:
        # Rk4 for 4th-order tensor
        f_i_jklm = (f_i_jklm_1 + 2 * f_i_jklm_2 + 2 * f_i_jklm_3 + f_i_jklm_4) / 6
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
        third_term_part1_4th = np.einsum("irsp,rjl,sk->ijklm", f_i_jkl, x_i_jk, x_i_j)
        third_term_part2_4th = np.einsum("irsp,rj,skl->ijklm", f_i_jkl, x_i_j, x_i_jk)
        third_term_part3_4th = np.einsum("irsp,rjk,sl->ijklm", f_i_jkl, x_i_jk, x_i_j)
        fourth_term_part1_4th = np.einsum("irs,rjlm,sk->ijklm", f_i_jk, x_i_jkl, x_i_j)
        fourth_term_part2_4th = np.einsum("irs,rjl,skm->ijklm", f_i_jk, x_i_jk, x_i_j)
        fourth_term_part3_4th = np.einsum("irs,rjm,skl->ijklm", f_i_jk, x_i_jk, x_i_j)
        fourth_term_part4_4th = np.einsum("irs,rk,sklm->ijklm", f_i_jk, x_i_j, x_i_jkl)
        fourth_term_part5_4th = np.einsum("irs,rjkm,sl->ijklm", f_i_jk, x_i_jkl, x_i_j)
        fourth_term_part6_4th = np.einsum("irs,rjk,slm->ijklm", f_i_jk, x_i_jk, x_i_j)
        fourth_term_part7_4th = np.einsum("irs,rjkl,sm->ijklm", f_i_jk, x_i_jkl, x_i_j)
        fifth_term_4th = np.einsum("ir,rjklm->ijklm", f_i_j, x_i_jklm)

        next_x_i_jklm = (
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
        ) * dt + x_i_jklm

    else:
        next_x_i_jklm = x_i_jklm

    return next_x, next_x_i_j, next_x_i_jk, next_x_i_jkl, next_x_i_jklm


H = 100e3
r0 = np.array([6371e3 + H, 0, 0])
v_circle = np.sqrt(GM_Earth / np.linalg.norm(r0))
ecc = 0.0
sma = np.linalg.norm(r0) / (1 - ecc)
v_p = np.sqrt(GM_Earth * (2 / np.linalg.norm(r0) - 1 / sma))
v0 = np.array([0, 0, 1]) * v_p


state0 = np.concatenate((r0, v0))

sst1_j0 = np.zeros((6, len(wrt)))
np.fill_diagonal(sst1_j0, 1)
sst1_k0 = np.zeros((6, len(wrt)))
sst2_jk0 = np.zeros((6, len(wrt), len(wrt)))
sst3_jkl0 = np.zeros((6, len(wrt), len(wrt), len(wrt)))
# sst4_jklm0 = np.zeros((6, len(wrt), len(wrt), len(wrt), len(wrt)))

# Initialize
sst1_j = sst1_j0
sst2_jk = sst2_jk0
sst3_jkl = sst3_jkl0
sst4_jklm = None


t_orbit = 2 * np.pi * np.sqrt(sma ** 3 / GM_Earth)
t = 0
dt = 100
trajectory = []
relevant_keys = ["x", "y", "z", "vx", "vy", "vz"]
t_end = t_orbit * 1.5

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
        eom, sym_eom, state, sst1_j, sst2_jk, sst3_jkl, sst4_jklm, dt, at, wrt, order=3
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
v_lim = 100
r_lim = 1000
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

# dx0[3:, :] = np.array([SX.flatten(), SY.flatten(), SZ.flatten()])
# dx0[:3, :] = np.array([SVX.flatten(), SVY.flatten(), SVZ.flatten()])

dx0 = np.random.multivariate_normal(
    np.zeros(6),
    np.diag([sigma_pos, sigma_pos, sigma_pos, sigma_vel, sigma_vel, sigma_vel]),
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
# dx_4 = np.einsum("ijklm,js,ks,ls,ms->is", sst4_jklm, dx0, dx0, dx0, dx0) / 24.0 + dx_3


# List to hold all perturbed trajectories
perturbed_trajectories = []

# print(dx0)

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
    # Initialize subplots
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("QtAgg")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]

    ax.scatter(
        dx0[axis_1, :],
        dx0[axis_2, :],
        c="black",
        s=10,
        label="Initial Perturbations",
        alpha=0.5,
    )
    ax.scatter(
        dx_1[axis_1, :],
        dx_1[axis_2, :],
        c="red",
        marker="x",
        s=10,
        label="SST1 Transformed",
        alpha=0.5,
    )
    ax.scatter(
        dx_2[axis_1, :], dx_2[axis_2, :], c="blue", s=4, label="SST2 Transformed", alpha=0.5
    )
    ax.scatter(
        dx_3[axis_1, :],
        dx_3[axis_2, :],
        c="green",
        s=1,
        label="SST3 Transformed",
        alpha=0.5,
    )
    ax.scatter(
        numerical_dx[:, axis_1],
        numerical_dx[:, axis_2],
        c="purple",
        marker="s",
        s=10,
        label="Numerically Perturbed",
        alpha=0.5,
    )

    ax.axis("equal")
    ax.set_xlabel(f"{axis_1_name.upper()} Position")
    ax.set_ylabel(f"{axis_2_name.upper()} Position")
    ax.legend()
    ax.set_title("Position Perturbations")

    # Velocity Perturbations
    ax = axes[1]
    ax.scatter(
        dx0[3 + axis_1, :],
        dx0[3 + axis_2, :],
        c="black",
        s=10,
        label="Initial Perturbations",
        alpha=0.5,
    )
    ax.scatter(
        dx_1[3 + axis_1, :],
        dx_1[3 + axis_2, :],
        c="red",
        marker="x",
        s=10,
        label="SST1 Transformed",
        alpha=0.5,
    )
    ax.scatter(
        dx_2[3 + axis_1, :],
        dx_2[3 + axis_2, :],
        c="blue",
        s=4,
        label="SST2 Transformed",
        alpha=0.5,
    )
    ax.scatter(
        dx_3[3 + axis_1, :],
        dx_3[3 + axis_2, :],
        c="green",
        s=1,
        label="SST3 Transformed",
        alpha=0.5,
    )
    ax.scatter(
        numerical_dx[:, 3 + axis_1],
        numerical_dx[:, 3 + axis_2],
        c="purple",
        marker="s",
        s=10,
        label="Numerically Perturbed",
        alpha=0.5,
    )
    ax.axis("equal")
    ax.set_xlabel(f"{axis_1_name.upper()} Velocity")
    ax.set_ylabel(f"{axis_2_name.upper()} Velocity")
    ax.legend()
    ax.set_title("Velocity Perturbations")

    # Full Trajectory with Position Perturbations
    ax = axes[2]
    ax.plot(trajectory[:, axis_1], trajectory[:, axis_2], "k-", label="Trajectory")
    ax.scatter(
        trajectory[-1, axis_1] + dx_1[axis_1, :],
        trajectory[-1, axis_2] + dx_1[axis_2, :],
        c="red",
        marker="x",
        s=10,
        label="SST1 Transformed",
        alpha=0.5,
    )
    ax.scatter(
        trajectory[-1, axis_1] + dx_2[axis_1, :],
        trajectory[-1, axis_2] + dx_2[axis_2, :],
        c="blue",
        s=4,
        label="SST2 Transformed",
        alpha=0.5,
    )
    ax.scatter(
        trajectory[-1, axis_1] + dx_3[axis_1, :],
        trajectory[-1, axis_2] + dx_3[axis_2, :],
        c="green",
        s=1,
        label="SST3 Transformed",
        alpha=0.5,
    )
    ax.scatter(
        final_perturbed_states[:, axis_1],
        final_perturbed_states[:, axis_2],
        c="purple",
        marker="s",
        s=10,
        label="Numerically Perturbed",
        alpha=0.5,
    )

    ax.set_aspect("equal")
    ax.set_xlabel(f"{axis_1_name.upper()} Position")
    ax.set_ylabel(f"{axis_2_name.upper()} Position")
    ax.legend()
    ax.set_title("Full Trajectory with Position Perturbations")

    plt.tight_layout()
    plt.show()

elif plot_with=="mayavi":

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