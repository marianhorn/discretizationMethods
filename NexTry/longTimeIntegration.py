import numpy as np
import matplotlib.pyplot as plt
from math import pi, log2
from RungeKutta import rk4_solver_matrix


def convergence_study():
    N_vals = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    T = np.pi
    dt = 0.001
    precision_digits = 50
    methods = ['fd2', 'fd4', 'fourier']
    errors = {method: [] for method in methods}
    rates = {method: [] for method in methods}

    for method in methods:
        print(f"\n--- Running convergence study for method: {method} ---")
        prev_error = None

        for N in N_vals:
            steps = int(round(T / dt))
            u_all, x = rk4_solver_matrix(N, dt, steps, method, precision_digits)
            u_num = u_all[:, -1]
            u_exact = np.exp(np.sin(x - 2 * np.pi * T))
            error = np.max(np.abs(u_num - u_exact))
            errors[method].append(error)

            # Estimate convergence rate
            if prev_error is not None:
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
    plt.title('Convergence of RK4 with different spatial discretizations at T = π')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

    # === Print convergence table ===
    print("\n--- Convergence Rate Summary ---")
    for method in methods:
        print(f"\nMethod: {method.upper()}")
        print(f"{'N':>6} | {'Error':>12} | {'Rate':>6}")
        print('-' * 30)
        for i in range(len(N_vals)):
            N = N_vals[i]
            err = errors[method][i]
            rate = rates[method][i - 1] if i > 0 else None
            print(f"{N:6d} | {err:12.4e} | {rate:6.2f}" if rate else f"{N:6d} | {err:12.4e} |   ---")


if __name__ == "__main__":
    convergence_study()
