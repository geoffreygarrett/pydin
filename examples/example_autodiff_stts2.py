import numpy as np
from sympy import Matrix, lambdify, diff, exp, symbols
import itertools


def compute_tensors(eom, wrt, at):
    m, n = len(eom), len(wrt)
    at_str = {str(k): v for k, v in at.items()}

    first_order = np.array(get_derivative(eom, wrt, 1, at)(**at_str)).reshape(m, n)
    second_order = np.array(get_derivative(eom, wrt, 2, at)(**at_str)).reshape(m, n, n)
    third_order = np.array(get_derivative(eom, wrt, 3, at)(**at_str)).reshape(m, n, n, n)
    # fourth_order = np.array(get_derivative(eom, wrt, 4, at)(**at_str)).reshape(m, n, n, n, n)

    # Using symmetry to optimize storage and computation for 2nd and 3rd order tensors
    # Fill only unique elements, copy the rest
    for j in range(n):
        for k in range(j + 1):
            second_order[:, k, j] = second_order[:, j, k]

    for j in range(n):
        for k in range(j + 1):
            for l in range(k + 1):
                third_order[:, j, l, k] = third_order[:, j, k, l]
                third_order[:, k, j, l] = third_order[:, j, k, l]
                third_order[:, k, l, j] = third_order[:, j, k, l]
                third_order[:, l, j, k] = third_order[:, j, k, l]
                third_order[:, l, k, j] = third_order[:, j, k, l]

    # for j in range(n):
    #     for k in range(j + 1):
    #         for l in range(k + 1):
    #             for m in range(l + 1):
    #                 fourth_order[:, j, k, l, m] = fourth_order[:, j, k, m, l]
    #                 fourth_order[:, j, k, m, l] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, j, l, k, m] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, j, l, m, k] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, j, m, k, l] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, j, m, l, k] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, k, j, l, m] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, k, j, m, l] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, k, l, j, m] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, k, l, m, j] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, k, m, j, l] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, k, m, l, j] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, l, j, k, m] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, l, j, m, k] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, l, k, j, m] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, l, k, m, j] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, l, m, j, k] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, l, m, k, j] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, m, j, k, l] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, m, j, l, k] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, m, k, j, l] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, m, k, l, j] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, m, l, j, k] = fourth_order[:, j, k, l, m]
    #                 fourth_order[:, m, l, k, j] = fourth_order[:, j, k, l, m]

    return first_order, second_order, third_order


derivative_cache = {}


def get_derivative(eom, wrt, order, at):
    cache_key = (tuple(eom), tuple(wrt), order, tuple(at.keys()))

    if cache_key in derivative_cache:
        return derivative_cache[cache_key]

    combinations = list(itertools.product(wrt, repeat=order))
    derivatives = []

    for eq in eom:
        eq_derivs = [diff(eq, *combination) for combination in combinations]
        derivatives.append(eq_derivs)

    # at_str = {str(k): v for k, v in at.items()}
    f_lambdified = lambdify(wrt, derivatives, modules="numpy")
    derivative_cache[cache_key] = f_lambdified

    return f_lambdified


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
rho0 = 3.8 * 10 ** -12


def rho_func(r):
    # res = rho0
    res = rho0 * np.exp(-(np.linalg.norm(r) - r0) / den)
    return res


def eom(state: np.ndarray, dt: float):
    rel_moon = state[:3] - r_moon
    rel_earth = state[:3] - r_earth
    accel_moon = -GM_Moon * rel_moon / (np.linalg.norm(rel_moon) ** 3)
    accel_earth = -GM_Earth * rel_earth / (np.linalg.norm(rel_earth) ** 3)
    accel_drag = -0.5 * rho_func(state[:3]) * np.linalg.norm(state[3:]) * CD * S / m * state[3:]

    # J2 Perturbation for Earth
    x, y, z = state[:3]
    r_norm = np.linalg.norm(state[:3])
    factor = 1.5 * J2 * (R_Earth / r_norm) ** 2
    j2_term_x = factor * (5 * (z / r_norm) ** 2 - 1) * x
    j2_term_y = factor * (5 * (z / r_norm) ** 2 - 1) * y
    j2_term_z = factor * (5 * (z / r_norm) ** 2 - 3) * z
    j2_accel = np.array([j2_term_x, j2_term_y, j2_term_z]) * GM_Earth / (r_norm ** 3)

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
sym_j2_accel = Matrix([sym_j2_term_x, sym_j2_term_y, sym_j2_term_z]) * GM_Earth / (r_norm ** 3)

# Previous accelerations
sym_accel = sym_accel_earth + sym_accel_moon + sym_j2_accel + sym_drag
sym_eom = Matrix([vx, vy, vz, sym_accel[0], sym_accel[1], sym_accel[2]])
wrt = [x, y, z, vx, vy, vz]


def rk4_step(func, sym_func, state, x_i_j, x_i_jk, x_i_jkl, dt, at, wrt):
    """Runge-Kutta 4th order method with tensor propagation."""

    def get_at(_state):
        return {**at, **{x: _state[0], y: _state[1], z: _state[2], vx: _state[3], vy: _state[4], vz: _state[5]}}

    # Standard RK4 for state
    k1 = func(state, dt)
    k2 = func(state + k1 / 2 * dt, dt)
    k3 = func(state + k2 / 2 * dt, dt)
    k4 = func(state + k3 * dt, dt)
    next_x = state + ((k1 + 2 * k2 + 2 * k3 + k4) / 6) * dt

    # Compute the Jacobian tensors at the RK4 stages
    f_i_j_1, f_i_jk_1, f_i_jkl_1 = compute_tensors(sym_func, wrt, get_at(state))
    f_i_j_2, f_i_jk_2, f_i_jkl_2 = compute_tensors(sym_func, wrt, get_at(state + k1 / 2 * dt))
    f_i_j_3, f_i_jk_3, f_i_jkl_3 = compute_tensors(sym_func, wrt, get_at(state + k2 / 2 * dt))
    f_i_j_4, f_i_jk_4, f_i_jkl_4 = compute_tensors(sym_func, wrt, get_at(state + k3 * dt))

    # RK4 for 1st-order tensor
    f_i_j = (f_i_j_1 + 2 * f_i_j_2 + 2 * f_i_j_3 + f_i_j_4) / 6
    next_x_i_j = x_i_j + np.einsum('ij,jk->ik', f_i_j, x_i_j) * dt

    # RK4 for 2nd-order tensor
    f_i_jk = (f_i_jk_1 + 2 * f_i_jk_2 + 2 * f_i_jk_3 + f_i_jk_4) / 6
    first_term = np.einsum('irs,rj,sk->ijk', f_i_jk, x_i_j, x_i_j)
    second_term = np.einsum('ir,rjk->ijk', f_i_j, x_i_jk)
    xdot_i_jk = first_term + second_term
    next_x_i_jk = x_i_jk + xdot_i_jk * dt

    # RK4 for 3rd-order tensor
    f_i_jkl = (f_i_jkl_1 + 2 * f_i_jkl_2 + 2 * f_i_jkl_3 + f_i_jkl_4) / 6

    # Calculating first_term = f_i_rsp * x_r_j * x_s_k * x_p_l
    first_term = np.einsum('irsp,rj,sk,pl->ijkl', f_i_jkl, x_i_j, x_i_j, x_i_j)

    # Calculating second_term
    # 𝑥_𝑖,𝑗𝑘𝑙=𝑓_𝑖,𝑟𝑠𝑝 𝑥_𝑟,𝑗 𝑥_𝑠,𝑘 𝑥_𝑝,𝑙 + 𝑓𝑖,𝑟𝑠(𝑥_𝑟,𝑗𝑙 𝑥_𝑠,𝑘 + 𝑥_𝑟,𝑗 𝑥_𝑠,𝑘𝑙 + 𝑥_𝑟,𝑗𝑘 𝑥_𝑠,𝑙)
    second_term_part1 = np.einsum('irs,rjl,sk->ijkl', f_i_jk, x_i_jk, x_i_j)
    second_term_part2 = np.einsum('irs,rj,skl->ijkl', f_i_jk, x_i_j, x_i_jk)
    second_term_part3 = np.einsum('irs,rjk,sl->ijkl', f_i_jk, x_i_jk, x_i_j)
    second_term = second_term_part1 + second_term_part2 + second_term_part3

    # Calculating third_term = f_i_r * x_r_jkl
    third_term = np.einsum('ir,rjkl->ijkl', f_i_j, x_i_jkl)

    # Combine all terms
    xdot_i_jkl = first_term + second_term + third_term
    next_x_i_jkl = x_i_jkl + xdot_i_jkl * dt

    return next_x, next_x_i_j, next_x_i_jk, next_x_i_jkl


H = 500e3
r0 = np.array([6371e3 + H, 0, 0])
v_circle = np.sqrt(GM_Earth / np.linalg.norm(r0))
v0 = np.array([0, 0, 1]) * v_circle
state = np.concatenate((r0, v0))

sst1_j0 = np.zeros((6, len(wrt)))
np.fill_diagonal(sst1_j0, 1)
sst1_k0 = np.zeros((6, len(wrt)))
sst2_jk0 = np.zeros((6, len(wrt), len(wrt)))
sst3_jkl0 = np.zeros((6, len(wrt), len(wrt), len(wrt)))
sst4_jklm0 = np.zeros((6, len(wrt), len(wrt), len(wrt), len(wrt)))

# Initialize
sst1_j = sst1_j0
sst2_jk = sst2_jk0
sst3_jkl = sst3_jkl0
t_orbit = 2 * np.pi * np.sqrt(np.linalg.norm(r0) ** 3 / GM_Earth)
t = 0
dt = 20
trajectory = []
relevant_keys = ['x', 'y', 'z', 'vx', 'vy', 'vz']
t_end = t_orbit * 10

while t < t_end:
    at = {x: state[0], y: state[1], z: state[2], vx: state[3], vy: state[4], vz: state[5]}

    # Update the next_state, next_sst1_j, next_sst2_jk
    state, sst1_j, sst2_jk, sst3_jkl = rk4_step(eom, sym_eom, state, sst1_j, sst2_jk, sst3_jkl0, dt, at, wrt)

    # Record the state
    trajectory.append(state)

    # Update the time
    t += dt

# Compute the terms using tensordot
print("sst1_j:", sst1_j)
print("sst2_jk:", sst2_jk)

import numpy as np
import matplotlib.pyplot as plt

# Simulate some data: Replace these with your actual calculations
# t_orbit, dt, state, GM_Earth, eom, dfdx, d2fdx2, wrt, sst1_j, sst2_jk would be defined here

# Initialize perturbations
n_perturbations = 500
n_s = 5j
v_lim = 100
r_lim = 1000
n_perturbations = int(5 ** 6)

# Grid sampling
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

dx0 = np.random.multivariate_normal(np.zeros(6),
                                    np.diag([sigma_pos,
                                             sigma_pos,
                                             sigma_pos,
                                             sigma_vel,
                                             sigma_vel,
                                             sigma_vel]),
                                    n_perturbations).T

# First order perturbations
dx_1 = np.einsum('ij,js->is', sst1_j, dx0)

# Second order; dimensions:
#  sst2_jk: (6, 6, 6)             (i, j, k)
#  1st dx0: (6, n_perturbations)  (j, s)
#  2nd dx0: (6, n_perturbations)  (k, s)
#  dx_1:    (6, n_perturbations)  (i, s)
dx_2 = np.einsum('ijk,js,ks->is', sst2_jk, dx0, dx0) / 2.0 + dx_1

# Third order
dx_3 = np.einsum('ijkl,js,ks,ls->is', sst3_jkl, dx0, dx0, dx0) / 6.0 + dx_2

# Initialize subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Position Perturbations
ax = axes[0]
axis_1 = 0
axis_2 = 2

axis_1_name = relevant_keys[axis_1]
axis_2_name = relevant_keys[axis_2]

ax.scatter(dx0[axis_1, :], dx0[axis_2, :], c='black', s=10, label='Initial Perturbations', alpha=0.5)
ax.scatter(dx_1[axis_1, :], dx_1[axis_2, :], c='red', marker='x', s=10, label='SST1 Transformed', alpha=0.5)
ax.scatter(dx_2[axis_1, :], dx_2[axis_2, :], c='blue', s=4, label='SST2 Transformed', alpha=0.5)
ax.scatter(dx_3[axis_1, :], dx_3[axis_2, :], c='green', s=1, label='SST3 Transformed', alpha=0.5)
ax.axis('equal')
ax.set_xlabel(f'{axis_1_name.upper()} Position')
ax.set_ylabel(f'{axis_2_name.upper()} Position')
ax.legend()
ax.set_title('Position Perturbations')

# Velocity Perturbations
ax = axes[1]
ax.scatter(dx0[3 + axis_1, :], dx0[3 + axis_2, :], c='black', s=10, label='Initial Perturbations', alpha=0.5)
ax.scatter(dx_1[3 + axis_1, :], dx_1[3 + axis_2, :], c='red', marker='x', s=10, label='SST1 Transformed', alpha=0.5)
ax.scatter(dx_2[3 + axis_1, :], dx_2[3 + axis_2, :], c='blue', s=4, label='SST2 Transformed', alpha=0.5)
ax.scatter(dx_3[3 + axis_1, :], dx_3[3 + axis_2, :], c='green', s=1, label='SST3 Transformed', alpha=0.5)
ax.axis('equal')
ax.set_xlabel(f'{axis_1_name.upper()} Velocity')
ax.set_ylabel(f'{axis_2_name.upper()} Velocity')
ax.legend()
ax.set_title('Velocity Perturbations')

# Full Trajectory with Position Perturbations
ax = axes[2]
trajectory = np.array(trajectory)
ax.plot(trajectory[:, axis_1], trajectory[:, axis_2], 'k-', label='Trajectory')
ax.scatter(trajectory[-1, axis_1] + dx_1[axis_1, :], trajectory[-1, axis_2] + dx_1[axis_2, :], c='red', marker='x',
           s=10,
           label='SST1 Transformed', alpha=0.5)
ax.scatter(trajectory[-1, axis_1] + dx_2[axis_1, :], trajectory[-1, axis_2] + dx_2[axis_2, :], c='blue', s=4,
           label='SST2 Transformed',
           alpha=0.5)
ax.scatter(trajectory[-1, axis_1] + dx_3[axis_1, :], trajectory[-1, axis_2] + dx_3[axis_2, :], c='green', s=1,
           label='SST3 Transformed',
           alpha=0.5)
ax.set_aspect('equal')
ax.set_xlabel(f'{axis_1_name.upper()} Position')
ax.set_ylabel(f'{axis_2_name.upper()} Position')
ax.legend()
ax.set_title('Full Trajectory with Position Perturbations')

plt.tight_layout()
plt.show()
