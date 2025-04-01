import numpy as np
import matplotlib.pyplot as plt
import gmpy2
from gmpy2 import mpfr, sin, exp, const_pi

# Set precision
gmpy2.get_context().precision = 256
pi = const_pi()

# Exact solution using gmpy2
def exact_solution(x, t):
    return np.array([float(exp(sin(mpfr(xi) - 2 * pi * mpfr(t)))) for xi in x], dtype=np.float64)

# Initial condition
def initial_condition(x):
    return exact_solution(x, 0)

# Second order finite difference derivative
def derivative_fd2(u, dx):
    return (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

# Fourth order finite difference derivative
def derivative_fd4(u, dx):
    return (-np.roll(u, 2) + 8 * np.roll(u, 1) - 8 * np.roll(u, -1) + np.roll(u, -2)) / (12 * dx)

# Fourier differentiation matrix using gmpy2 for high precision
def fourier_diff_matrix_gmp(N):
    D = [[mpfr(0) for _ in range(N)] for _ in range(N)]
    for j in range(N):
        for i in range(N):
            if i != j:
                angle = (j - i) * pi / mpfr(N)
                D[j][i] = ((-1)**(j + i)) / (2 * sin(angle))
    return D

# Matrix-vector multiplication using gmpy2 matrix
def matvec_gmp(D, u):
    N = len(u)
    result = [mpfr(0) for _ in range(N)]
    for j in range(N):
        for i in range(N):
            result[j] += D[j][i] * mpfr(u[i])
    return np.array([float(-2 * pi * r) for r in result], dtype=np.float64)

# Time stepping using 4th order Runge-Kutta
def rk4_step(u, dt, derivative_func):
    k1 = dt * derivative_func(u)
    k2 = dt * derivative_func(u + 0.5 * k1)
    k3 = dt * derivative_func(u + 0.5 * k2)
    k4 = dt * derivative_func(u + k3)
    return u + (k1 + 2*k2 + 2*k3 + k4) / 6

# Solver
def solve_pde(N, T, dt, method='fd2'):
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    dx = x[1] - x[0]
    u = initial_condition(x)

    if method == 'fd2':
        derivative = lambda u: -2 * np.pi * derivative_fd2(u, dx)
    elif method == 'fd4':
        derivative = lambda u: -2 * np.pi * derivative_fd4(u, dx)
    elif method == 'fourier':
        D = fourier_diff_matrix_gmp(N)
        derivative = lambda u: matvec_gmp(D, u)
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