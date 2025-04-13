import numpy as np
import matplotlib.pyplot as plt
from math import log2
from mpmath import mp, sin, exp, pi, mpf
from RungeKutta import rk4_solver_matrix  # <- your existing RK4 implementation


def convergence_study_high_precision():
    N_vals = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    T = mp.pi
    dt = mpf("0.001")
    precision_digits = 70
    mp.dps = precision_digits

    methods = ['fd2', 'fd4', 'fourier']
    errors = {method: [] for method in methods}
    rates = {method: [] for method in methods}

    for method in methods:
        print(f"\n--- Running convergence study for method: {method} ---")
        prev_error = None

        for N in N_vals:
            steps = int(mp.nint(T / dt))
            u_all, x = rk4_solver_matrix(N, float(dt), steps, method, precision_digits)
            u_num = u_all[:, -1]

            # Convert x to high-precision and compute exact solution
            x_mp = [mpf(xi) for xi in x]
            u_exact = np.array([float(exp(sin(xi - 2 * pi * T))) for xi in x_mp])
            error = float(np.max(np.abs(u_num - u_exact)))
            errors[method].append(error)

            # Compute convergence rate
            if prev_error is not None and error > 0:
                rate = log2(prev_error / error)
                rates[method].append(rate)
                print(f"N = {N:4d} | Error = {error:.3e} | Rate ≈ {rate:.2f}")
            else:
                print(f"N = {N:4d} | Error = {error:.3e}")
            prev_error = error

    # === Plotting ===
    plt.figure(figsize=(10, 6))
    for method in methods:
        plt.loglog(N_vals, errors[method], '-o', label=method.upper())

    plt.xlabel('N (grid points)')
    plt.ylabel('L∞ error')
    plt.title('High-Precision Convergence at T = π')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

    # === Print convergence rate table ===
    print("\n--- Convergence Rate Summary ---")
    for method in methods:
        print(f"\nMethod: {method.upper()}")
        print(f"{'N':>6} | {'Error':>12} | {'Rate':>6}")
        print('-' * 30)
        for i in range(len(N_vals)):
            N = N_vals[i]
            err = errors[method][i]
            if i > 0:
                rate = rates[method][i - 1]
                print(f"{N:6d} | {err:12.4e} | {rate:6.2f}")
            else:
                print(f"{N:6d} | {err:12.4e} |   ---")


if __name__ == "__main__":
    convergence_study_high_precision()
