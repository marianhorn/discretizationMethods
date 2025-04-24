import numpy as np
import matplotlib.pyplot as plt
from FP_RungeKutta import rk4_solver_matrix


def evaluate_long_time_transport_float64():
    dt = 0.001
    times = [0, 100, 200]

    schemes = [
        {"method": "fd2", "N": 200, "label": "2nd Order FD"},
        {"method": "fourier", "N": 10, "label": "Infinite Order (Fourier)"}
    ]

    for scheme in schemes:
        method = scheme["method"]
        N = scheme["N"]
        label = scheme["label"]

        print(f"\n>>> Starting scheme: {label} (method = '{method}', N = {N})")

        fig, axs = plt.subplots(1, len(times), figsize=(15, 4))
        fig.suptitle(f"{label} – Comparison at t = 0, 100, 200 (N = {N})")

        for i, T in enumerate(times):
            print(f"  → Evaluating at t = {T}")
            steps = int(round(T / dt))
            u_num, x = rk4_solver_matrix(N, dt, steps, method)

            u_exact = np.exp(np.sin(x - 2 * np.pi * T))

            axs[i].plot(x, u_num, 'bo-', label='Numerical')  # blue circles
            axs[i].plot(x, u_exact, 'r--x', label='Exact')    # red dashed line with x markers
            axs[i].set_title(f"t = {T}")
            axs[i].set_xlabel('x')
            axs[i].set_ylabel('u(x)')
            axs[i].grid(True)

        axs[0].legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    evaluate_long_time_transport_float64()
