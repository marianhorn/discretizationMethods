import numpy as np
import mpmath
import matplotlib.pyplot as plt


def rk4_solver_matrix(N, dt, steps, method='fourier', precision_digits=50):
    # === Grid and initial condition ===
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)  # Periodic grid
    u = np.exp(np.sin(x))
    u_all = np.zeros((N, steps + 1))
    u_all[:, 0] = u.copy()

    # === Setup differentiation operator ===
    if method == 'fourier':
        D, _ = fourier_diff_matrix_vpa(N, precision_digits, 'odd')
        D = np.array([[float(entry) for entry in row] for row in D])
    else:
        dx = x[1] - x[0]

    # === Time stepping ===
    for n in range(steps):
        def compute_rhs(u_in):
            if method == 'fd2':
                return -2 * np.pi * (np.roll(u_in, -1) - np.roll(u_in, 1)) / (2 * dx)
            elif method == 'fd4':
                return -2 * np.pi * (-np.roll(u_in, 2) + 8 * np.roll(u_in, 1)
                                     - 8 * np.roll(u_in, -1) + np.roll(u_in, -2)) / (12 * dx)
            elif method == 'fourier':
                return -2 * np.pi * D @ u_in
            else:
                raise ValueError("Unknown method")

        u1 = u + 0.5 * dt * compute_rhs(u)
        u2 = u + 0.5 * dt * compute_rhs(u1)
        u3 = u + dt * compute_rhs(u2)
        u = (1 / 3) * (-u + u1 + 2 * u2 + u3 + 0.5 * dt * compute_rhs(u3))

        u_all[:, n + 1] = u

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
        # Truncate matrix and grid to size N
        D = [row[:N] for row in D_full[:N]]
        x = x_full[:N]
        return D, x
    else:
        raise ValueError("Only 'odd' method is supported for Fourier")


# === Parameters ===
N = 128
L = 2 * np.pi
x = np.linspace(0, L, N, endpoint=False)
dx = x[1] - x[0]
dt = 0.001
T = 1.0
steps = round(T / dt)

method = 'fd4'  # 'fd2', 'fd4', or 'fourier'
precision_digits = 50

# === Initial condition ===
u0 = np.exp(np.sin(x))

# === Run RK4 solver ===
u_all, x = rk4_solver_matrix(N, dt, steps, method, precision_digits)
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
