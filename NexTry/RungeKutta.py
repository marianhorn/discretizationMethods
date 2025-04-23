import numpy as np
import mpmath
import matplotlib.pyplot as plt


def rk4_solver_matrix(N, dt, steps, method='fourier', precision_digits=50):
    mpmath.mp.dps = precision_digits
    dt = mpmath.mpf(dt)

    if method == 'fourier':
        N_full = N + 1
        D, x = fourier_diff_matrix_vpa(N_full, precision_digits, 'odd')
        D = [[mpmath.mpf(dij) for dij in row] for row in D]
        u = [mpmath.exp(mpmath.sin(xi)) for xi in x]
        u_all = [[ui] for ui in u]

        def compute_rhs(u_in):
            return [-2 * mpmath.pi * sum(D[i][j] * u_in[j] for j in range(N_full)) for i in range(N_full)]

        N_used = N_full  # Use full size for iteration

    elif method in ['fd2', 'fd4']:
        L = mpmath.mpf(2) * mpmath.pi
        x = [L * i / N for i in range(N)]
        u = [mpmath.exp(mpmath.sin(xi)) for xi in x]
        u_all = [[ui] for ui in u]
        dx = x[1] - x[0]

        def compute_rhs(u_in):
            rhs = []
            for i in range(N):
                if method == 'fd2':
                    val = (-2 * mpmath.pi *
                           (u_in[(i + 1) % N] - u_in[(i - 1) % N]) / (2 * dx))
                elif method == 'fd4':
                    val = (-2 * mpmath.pi *
                           (-u_in[(i + 2) % N] + 8 * u_in[(i + 1) % N]
                            - 8 * u_in[(i - 1) % N] + u_in[(i - 2) % N]) / (12 * dx))
                rhs.append(val)
            return rhs

        N_used = N

    else:
        raise ValueError("Unknown method")

    for _ in range(steps):
        F = compute_rhs(u)
        u1 = [u[i] + mpmath.mpf('0.5') * dt * F[i] for i in range(N_used)]
        F1 = compute_rhs(u1)
        u2 = [u[i] + mpmath.mpf('0.5') * dt * F1[i] for i in range(N_used)]
        F2 = compute_rhs(u2)
        u3 = [u[i] + dt * F2[i] for i in range(N_used)]
        F3 = compute_rhs(u3)

        u = [(mpmath.mpf('1') / 3) * (-u[i] + u1[i] + 2 * u2[i] + u3[i] + mpmath.mpf('0.5') * dt * F3[i])
             for i in range(N_used)]
        for i in range(N_used):
            u_all[i].append(u[i])

    u_all = np.array([[float(val) for val in row] for row in u_all])
    x = np.array([float(xi) for xi in x])
    return u_all, x


def fourier_diff_matrix_vpa(N, precision_digits, method):
    mpmath.mp.dps = precision_digits
    if method == 'odd':
        h = mpmath.mpf(2) * mpmath.pi / N
        x = [h * i for i in range(N)]
        D = [[mpmath.mpf(0) for _ in range(N)] for _ in range(N)]
        for j in range(N):
            for i in range(N):
                if i != j:
                    D[j][i] = (-1) ** (i + j) / (2 * mpmath.sin((j - i) * mpmath.pi / N))
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
    precision_digits = 50

    # === Run RK4 solver ===
    u_all, x = rk4_solver_matrix(N, dt, steps, method, precision_digits)
    u_final = u_all[:, -1]

    # === Exact solution at t = T ===
    x_mp = [mpmath.mpf(xi) for xi in x]
    u_exact = np.array([float(mpmath.exp(mpmath.sin(xi - 2 * mpmath.pi * T))) for xi in x_mp])

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
