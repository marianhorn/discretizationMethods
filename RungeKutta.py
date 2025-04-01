import numpy as np
import matplotlib.pyplot as plt

# Exact solution
def exact_solution(x, t):
    return np.exp(np.sin(x - 2 * np.pi * t))

# Initial condition
def initial_condition(x):
    return exact_solution(x, 0)

# Second order finite difference derivative
def derivative_fd2(u, dx):
    return (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

# Fourth order finite difference derivative
def derivative_fd4(u, dx):
    return (-np.roll(u, 2) + 8 * np.roll(u, 1) - 8 * np.roll(u, -1) + np.roll(u, -2)) / (12 * dx)

# Fourier differentiation matrix
def fourier_diff_matrix(N):
    D = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            if i != j:
                D[j, i] = (-1)**(j + i) / (2 * np.sin((j - i) * np.pi / N))
    return D

# Time stepping using 4th order Runge-Kutta
def rk4_step(u, dt, derivative_func):
    k1 = dt * derivative_func(u)
    k2 = dt * derivative_func(u + 0.5 * k1)
    k3 = dt * derivative_func(u + 0.5 * k2)
    k4 = dt * derivative_func(u + k3)
    return u + (k1 + 2*k2 + 2*k3 + k4) / 6

# Solver
def solve_pde(N, T, dt, method='fd2'):
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    dx = x[1] - x[0]
    u = initial_condition(x)

    if method == 'fd2':
        derivative = lambda u: -2 * np.pi * derivative_fd2(u, dx)
    elif method == 'fd4':
        derivative = lambda u: -2 * np.pi * derivative_fd4(u, dx)
    elif method == 'fourier':
        D = fourier_diff_matrix(N)
        derivative = lambda u: -2 * np.pi * D @ u
    else:
        raise ValueError("Unknown method")

    t = 0
    while t < T:
        u = rk4_step(u, dt, derivative)
        t += dt
    return x, u, exact_solution(x, T)

# Compute L-infinity error
def compute_error(u_num, u_exact):
    return np.max(np.abs(u_num - u_exact))

# Convergence test
if __name__ == '__main__':
    N_values = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    methods = ['fd2', 'fd4', 'fourier']
    T = np.pi
    dt = 0.001

    for method in methods:
        print(f"\nMethod: {method}")
        errors = []
        for N in N_values:
            x, u_num, u_exact = solve_pde(N, T, dt, method)
            error = compute_error(u_num, u_exact)
            errors.append(error)
            print(f"N = {N:<5} Error = {error:.3e}")

        # Estimate convergence rates
        rates = [np.log(errors[i]/errors[i-1]) / np.log(N_values[i]/N_values[i-1]) for i in range(1, len(errors))]
        print("Convergence rates:")
        for i, rate in enumerate(rates):
            print(f"N = {N_values[i+1]:<5} Rate ≈ {rate:.2f}")