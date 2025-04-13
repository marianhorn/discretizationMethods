import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpf, sin, exp, pi
from RungeKutta import rk4_solver_matrix


def evaluate_long_time_transport():
    # === High precision setup ===
    precision_digits = 70
    mp.dps = precision_digits
    dt = mpf("0.001")
    times = [0, 100, 200]

    # === Two configurations to compare ===
    schemes = [
        {"method": "fd2", "N": 200, "label": "2nd Order FD"},
        {"method": "fourier", "N": 10, "label": "Infinite Order (Fourier)"}
    ]

    for scheme in schemes:
        method = scheme["method"]
        N = scheme["N"]
        label = scheme["label"]

        fig, axs = plt.subplots(1, len(times), figsize=(15, 4))
        fig.suptitle(f"{label} – Comparison at t = 0, 100, 200 (N = {N})")

        for i, T in enumerate(times):
            steps = int(mp.nint(T / dt))
            u_all, x = rk4_solver_matrix(N, float(dt), steps, method, precision_digits)
            x_mp = [mpf(xi) for xi in x]

            # Compute exact solution
            u_exact = np.array([float(exp(sin(xi - 2 * pi * T))) for xi in x_mp])
            u_num = u_all[:, -1]

            # Plotting
            axs[i].plot(x, u_num, 'b-', label='Numerical')
            axs[i].plot(x, u_exact, 'r--', label='Exact')
            axs[i].set_title(f"t = {T}")
            axs[i].set_xlabel('x')
            axs[i].set_ylabel('u(x)')
            axs[i].grid(True)

        axs[0].legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    evaluate_long_time_transport()
