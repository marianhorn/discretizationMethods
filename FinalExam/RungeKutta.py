import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, exp


def build_fourier_diff_matrix(N):
    D = np.zeros((N + 1, N + 1), dtype=np.float64)
    for j in range(N + 1):
        for i in range(N + 1):
            if i != j:
                sign = (-1) ** (j + i)
                angle = (j - i) * pi / (N + 1)
                D[j, i] = sign / (2 * sin(angle))
    return D


def rk4_burgers_matrix(N, CFL, T):
    N_full = N + 1
    h = 2 * pi / N_full
    x = np.array([j * h for j in range(N_full)])

    # Build D and D^2 matrices
    D = build_fourier_diff_matrix(N)
    D2 = D @ D

    # Initial condition at t = 0: u(x, 0)
    phi = np.zeros(N_full)
    dphi = np.zeros(N_full)
    for j in range(N_full):
        xj = x[j]
        for m in range(-50, 51):
            shift = xj - (2 * m + 1) * pi
            phi[j] += np.exp(-shift ** 2 / (4 * nu))
            dphi[j] += shift * np.exp(-shift ** 2 / (4 * nu)) / (2 * nu)
    u = c - 2 * nu * dphi / phi

    # Stable timestep using CFL condition
    dx = h
    dt = CFL / (np.max(np.abs(u) / dx + nu / dx ** 2))
    steps = int(round(T / dt))
    print("Final time simulation: ")
    print(dt*steps)
    print("Exact time:")
    print(T)
    def compute_rhs(u_vec):
        u_x = D @ u_vec
        u_xx = D2 @ u_vec
        return -u_vec * u_x + nu * u_xx

    # RK4 time stepping
    for _ in range(steps):
        F = compute_rhs(u)
        u1 = u + 0.5 * dt * F
        F1 = compute_rhs(u1)
        u2 = u + 0.5 * dt * F1
        F2 = compute_rhs(u2)
        u3 = u + dt * F2
        F3 = compute_rhs(u3)
        u = (1.0 / 3.0) * (-u + u1 + 2 * u2 + u3 + 0.5 * dt * F3)

    return x, u, dt, steps


def exact_solution(x, t, nu, c):
    phi = np.zeros_like(x)
    dphi = np.zeros_like(x)
    b = t + 1
    for m in range(-50, 51):
        shift = x - c * t - (2 * m + 1) * pi
        exp_term = np.exp(-shift ** 2 / (4 * nu * b))
        phi += exp_term
        dphi += shift * exp_term / (2 * nu * b)
    return c - 2 * nu * dphi / phi



def main():
    global nu, c
    N = 64
    CFL = 0.01
    T = np.pi/4
    nu = 0.1
    c = 4.0

    x, u_num, dt, steps = rk4_burgers_matrix(N, CFL, T)
    T_sim = dt*steps
    u_exact = exact_solution(x, T_sim, nu, c)

    error_Linf = np.max(np.abs(u_num - u_exact))
    error_L2 = np.sqrt(np.mean((u_num - u_exact) ** 2))

    print('--- RK4 Burgers with Matrix Spectral Derivatives (float64) ---')
    print(f'N (modes): {N}')
    print(f'CFL: {CFL}')
    print(f'Time: t = {T_sim:.4f}')
    print(f'L-inf error: {error_Linf:.3e}')
    print(f'L2 error:    {error_L2:.3e}')

    plt.figure(figsize=(10, 5))
    plt.plot(x, u_num, label='Numerical')
    plt.plot(x, u_exact, '--', label='Exact')
    plt.title(f'Burgers\' Equation at t = {T:.2f}, N = {N + 1}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
