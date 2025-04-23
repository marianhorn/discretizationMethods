import numpy as np
import matplotlib.pyplot as plt


def rk4_solver_matrix(N, dt, steps, method='fourier'):
    if method == 'fourier':
        N_full = N + 1
        D, x = fourier_diff_matrix_float64(N_full, 'odd')
        D = np.array(D)
        x = np.array(x)
        u = np.exp(np.sin(x))
        u_all = np.zeros((N_full, steps + 1))
        u_all[:, 0] = u
        N_used = N_full

        def compute_rhs(u_in):
            return -2 * np.pi * D @ u_in

    elif method in ['fd2', 'fd4']:
        L = 2 * np.pi
        x = np.linspace(0, L, N, endpoint=False)
        u = np.exp(np.sin(x))
        u_all = np.zeros((N, steps + 1))
        u_all[:, 0] = u
        dx = x[1] - x[0]
        N_used = N

        def compute_rhs(u_in):
            if method == 'fd2':
                return -2 * np.pi * (np.roll(u_in, -1) - np.roll(u_in, 1)) / (2 * dx)
            elif method == 'fd4':
                return -2 * np.pi * (-np.roll(u_in, 2) + 8 * np.roll(u_in, 1)
                                     - 8 * np.roll(u_in, -1) + np.roll(u_in, -2)) / (12 * dx)

    else:
        raise ValueError("Unknown method")

    for n in range(steps):
        F = compute_rhs(u)
        u1 = u + 0.5 * dt * F
        F1 = compute_rhs(u1)
        u2 = u + 0.5 * dt * F1
        F2 = compute_rhs(u2)
        u3 = u + dt * F2
        F3 = compute_rhs(u3)

        u = (1.0 / 3.0) * (-u + u1 + 2 * u2 + u3 + 0.5 * dt * F3)
        u_all[:, n + 1] = u

    return u_all, x


def fourier_diff_matrix_float64(N, method):
    if method == 'odd':
        h = 2 * np.pi / N
        x = np.array([h * i for i in range(N)])
        D = np.zeros((N, N))
        for j in range(N):
            for i in range(N):
                if i != j:
                    D[j, i] = (-1) ** (i + j) / (2 * np.sin((j - i) * np.pi / N))
        return D, x
    else:
        raise ValueError("Only 'odd' method is supported for Fourier")


def main():
    # === Parameters ===
    N = 64
    dt = 0.001
    T = 2.0
    steps = round(T / dt)
    method = 'fourier'  # 'fd2', 'fd4', or 'fourier'

    # === Run RK4 solver ===
    u_all, x = rk4_solver_matrix(N, dt, steps, method)
    u_final = u_all[:, -1]

    # === Exact solution at t = T ===
    u_exact = np.exp(np.sin(x - 2 * np.pi * T))

    # === Error metrics ===
    error_Linf = np.max(np.abs(u_final - u_exact))
    error_L2 = np.sqrt(np.mean((u_final - u_exact) ** 2))

    print('--- RK4 Evaluation ---')
    print(f'Method: {method}')
    print(f'Grid points N: {N}')
    print(f'Final time T: {T:.2f}')
    print(f'L-infinity error: {error_Linf:.3e}')
    print(f'L2 error:         {error_L2:.3e}')

    # === Plotting ===
    plt.figure(figsize=(10, 5))
    plt.plot(x, u_final, 'b-', label='Numerical')
    plt.plot(x, u_exact, 'r--', label='Exact')
    plt.title(f'RK4 Solution vs Exact at t = {T:.2f}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
