import numpy as np
import matplotlib.pyplot as plt
from RKcollocation import solve_burgers
# Task 2d
def main():
    N = 128
    CFL = 0.8
    times = [0, np.pi / 8, np.pi / 6, np.pi / 4]
    labels = [r"$t = 0$", r"$t = \pi/8$", r"$t = \pi/6$", r"$t = \pi/4$"]

    plt.figure(figsize=(8, 6))

    for t, label in zip(times, labels):
        x, u, _ = solve_burgers(CFL=CFL, N=N+1, t_final=t, nu=0.1, c=4.0)
        plt.plot(x, u, label=label)

    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title(f"Solution of Burgers' Equation for N = {N}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
