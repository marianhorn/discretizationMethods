import numpy as np
import matplotlib.pyplot as plt
from math import log2, pi
from joblib import Parallel, delayed
from FP_RungeKutta import rk4_solver_matrix


def simulate_one_case(N, T, dt, method):
    steps = int(round(T / dt))
    u_num, x = rk4_solver_matrix(N, dt, steps, method)
    u_exact = np.exp(np.sin(x - 2 * np.pi * T))
    error = np.max(np.abs(u_num - u_exact))
    return N, error


def convergence_study_float64():
    N_vals_full = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    T = np.pi
    dt = 1e-6
    methods = ['fd2', 'fd4', 'fourier']

    for method in methods:
        print(f"\n--- Running parallel convergence study for method: {method} ---")

        # Skip large Ns for Fourier

        N_vals = N_vals_full

        results = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(simulate_one_case)(N, T, dt, method) for N in N_vals
        )

        results.sort()
        N_list, error_list = zip(*results)
        rates = [None]
        for i in range(1, len(N_list)):
            rate = log2(error_list[i - 1] / error_list[i])
            rates.append(rate)

        plt.loglog(N_list, error_list, '-o', label=method.upper())

        print(f"\nMethod: {method.upper()}")
        print(f"{'N':>6} | {'Error':>12} | {'Rate':>6}")
        print('-' * 30)
        for N, err, rate in zip(N_list, error_list, rates):
            if rate is not None:
                print(f"{N:6d} | {err:12.4e} | {rate:6.2f}")
            else:
                print(f"{N:6d} | {err:12.4e} |   ---")

    plt.xlabel('N (grid points)')
    plt.ylabel('L∞ error')
    plt.title('RK4 Convergence at T = π (float64)')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    convergence_study_float64()
