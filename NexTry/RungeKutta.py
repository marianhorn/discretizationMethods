import numpy as np
import mpmath
import matplotlib.pyplot as plt


def rk4_solver_matrix(N, dt, steps, method='fourier', precision_digits=50):
    mpmath.mp.dps = precision_digits
    dt = mpmath.mpf(dt)

    L = mpmath.mpf(2) * mpmath.pi
    x = [L * i / N for i in range(N)]
    u = [mpmath.exp(mpmath.sin(xi)) for xi in x]
    u_all = [[ui] for ui in u]

    if method == 'fourier':
        D, _ = fourier_diff_matrix_vpa(N, precision_digits, 'odd')
        D = [[mpmath.mpf(dij) for dij in row] for row in D]

        def compute_rhs(u_in):
            return [-2 * mpmath.pi * sum(D[i][j] * u_in[j] for j in range(N)) for i in range(N)]

    elif method in ['fd2', 'fd4']:
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

    else:
        raise ValueError("Unknown method")

    for _ in range(steps):
        F = compute_rhs(u)
        u1 = [u[i] + mpmath.mpf('0.5') * dt * F[i] for i in range(N)]
        F1 = compute_rhs(u1)
        u2 = [u[i] + mpmath.mpf('0.5') * dt * F1[i] for i in range(N)]
        F2 = compute_rhs(u2)
        u3 = [u[i] + dt * F2[i] for i in range(N)]
        F3 = compute_rhs(u3)

        u = [(mpmath.mpf('1') / 3) * (-u[i] + u1[i] + 2 * u2[i] + u3[i] + mpmath.mpf('0.5') * dt * F3[i])
             for i in range(N)]
        for i in range(N):
            u_all[i].append(u[i])

    u_all = np.array([[float(val) for val in row] for row in u_all])
    x = np.array([float(xi) for xi in x])
    return u_all, x


def fourier_diff_matrix_vpa(N, precision_digits, method):
    mpmath.mp.dps = precision_digits
    if method == 'odd':
        Nsym = N + 1
        h = mpmath.mpf(2) * mpmath.pi / Nsym
        x_full = [h * i for i in range(Nsym)]
        D_full = [[mpmath.mpf(0) for _ in range(Nsym)] for _ in range(Nsym)]
        for j in range(Nsym):
            for i in range(Nsym):
                if i != j:
                    D_full[j][i] = (-1) ** (i + j) / (2 * mpmath.sin((j - i) * mpmath.pi / Nsym))
        D = [row[:N] for row in D_full[:N]]
        x = x_full[:N]
        return D, x
    else:
        raise ValueError("Only 'odd' method is supported for Fourier")


def main():
    # === Parameters ===
    N = 128
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
