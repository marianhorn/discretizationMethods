import numpy as np
import matplotlib.pyplot as plt


def solve_burgers(CFL, N, t_final, nu=0.1, c=4.0, terms=50):

    assert N % 2 == 1, "N must be odd"
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    dx = L / N

    def fourier_collocation_matrix(x):
        N = len(x)
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = (-1) ** (i + j) / (2 * np.sin((x[i] - x[j]) / 2))
        return D

    D = fourier_collocation_matrix(x)
    D2 = D @ D

    def phi(a, b, terms=terms):
        sum_phi = np.zeros_like(a)
        for k in range(-terms, terms + 1):
            sum_phi += np.exp(-((a - (2 * k + 1) * np.pi) ** 2) / (4 * nu * b))
        return sum_phi

    def dphi_dx_analytical(a, b, nu, terms):
        result = np.zeros_like(a)
        for k in range(-terms, terms + 1):
            shift = a - (2 * k + 1) * np.pi
            result += (-shift / (2 * nu * b)) * np.exp(-shift ** 2 / (4 * nu * b))
        return result
    def analytic_solution(x, t):
        a = x - c * t
        b = t + 1
        phi_val = phi(a, b)
        dphi_val = dphi_dx_analytical(a, b, nu, terms)
        phi_val = np.maximum(phi_val, 1e-14)
        return c - 2 * nu * dphi_val / phi_val

    u = analytic_solution(x, 0)

    def F(u):
        return -u * (D @ u) + nu * (D2 @ u)

    def compute_dt(u):
        umax = np.max(np.abs(u))
        return CFL / (umax / dx + nu / dx ** 2)

    t = 0.0
    dt = compute_dt(u)

    while t < t_final:
        if t + dt > t_final:
            dt = t_final - t

        F0 = F(u)
        u1 = u + 0.5 * dt * F0
        F1 = F(u1)
        u2 = u + 0.5 * dt * F1
        F2 = F(u2)
        u3 = u + dt * F2
        F3 = F(u3)

        u = (1 / 3) * (-u + u1 + 2 * u2 + u3 + 0.5 * dt * F3)
        t += dt
        dt = compute_dt(u)

    u_exact = analytic_solution(x, t_final)
    return x, u, u_exact


def main():
    CFL = 0.8
    N = 129
    t_final = 0.1
    x, u, u_exact = solve_burgers(CFL, N, t_final)

    plt.plot(x, u, label='Numerical', linewidth=2)
    plt.plot(x, u_exact, '--', label='Analytical', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title(f"Burgers' Equation at t = {t_final}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
