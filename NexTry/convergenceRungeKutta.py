import numpy as np
import matplotlib.pyplot as plt
from math import log2
from mpmath import mp, mpf, sin, exp, pi
from joblib import Parallel, delayed
from RungeKutta import rk4_solver_matrix


def simulate_one_case(N, T_float, dt_float, method, precision_digits):
    mp.dps = precision_digits
    T = mpf(T_float)
    dt = mpf(dt_float)

    steps = int(mp.nint(T / dt))
    u_all, x = rk4_solver_matrix(N, float(dt), steps, method, precision_digits)
    x_mp = [mpf(xi) for xi in x]
    u_num = u_all[:, -1]
    u_exact = np.array([float(exp(sin(xi - 2 * pi * T))) for xi in x_mp])
    error = float(np.max(np.abs(u_num - u_exact)))
    return N, error


def convergence_study_high_precision_parallel():
   # N_vals = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    N_vals = [8, 16, 32, 64, 128]

    T_float = float(mp.pi)
    dt_float = 0.0001
    precision_digits = 15
    mp.dps = precision_digits

   # methods = ['fd2', 'fd4', 'fourier']
    methods = [ 'fd2','fourier']
    for method in methods:
        print(f"\n--- Running parallel convergence study for method: {method} ---")

        results = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(simulate_one_case)(N, T_float, dt_float, method, precision_digits)
            for N in N_vals
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
    plt.title('High-Precision RK4 Convergence at T = π')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    convergence_study_high_precision_parallel()
