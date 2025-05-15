import numpy as np
import matplotlib.pyplot as plt

def solve_burgers_galerkin(CFL, N, t_final, nu=0.1, c=4.0):
    # Grid
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    L = 2 * np.pi
    k = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    k = 1j * k
    k2 = -k ** 2
    kmax = N // 2

    # Initial condition: analytical expression at t = 0
    def phi(a, b, terms=50):
        return sum(np.exp(-((a - (2*k + 1)*np.pi)**2) / (4*nu*b)) for k in range(-terms, terms+1))

    def analytic_u(x, t):
        a = x - c * t
        b = t + 1
        φ = phi(a, b)
        dφ = (phi(a + 1e-6, b) - phi(a - 1e-6, b)) / (2e-6)
        return c - 2 * nu * dφ / φ

    u = analytic_u(x, 0)
    u_hat = np.fft.fft(u)

    def nonlinear_term(u_hat):
        u = np.fft.ifft(u_hat)
        ux = np.fft.ifft(k * u_hat)
        nonlin = -u * ux
        return np.fft.fft(nonlin)

    def rhs(u_hat):
        return nonlinear_term(u_hat) + nu * (k2 * u_hat)

    def compute_dt(u):
        umax = max(np.max(np.abs(u)), 1e-8)
        return CFL / (umax * kmax + nu * kmax**2)

    t = 0.0
    u = np.fft.ifft(u_hat)
    dt = compute_dt(u)

    while t < t_final:
        if t + dt > t_final:
            dt = t_final - t

        F0 = rhs(u_hat)
        u1 = u_hat + 0.5 * dt * F0
        F1 = rhs(u1)
        u2 = u_hat + 0.5 * dt * F1
        F2 = rhs(u2)
        u3 = u_hat + dt * F2
        F3 = rhs(u3)

        u_hat = (1/3) * (-u_hat + u1 + 2*u2 + u3 + 0.5 * dt * F3)
        t += dt
        dt = compute_dt(np.fft.ifft(u_hat))

    u_final = np.real(np.fft.ifft(u_hat))
    u_exact = analytic_u(x, t_final)
    return x, u_final, u_exact

if __name__ == "__main__":
    CFL = 0.1
    N = 32
    t_final = np.pi / 16

    x, u, u_exact = solve_burgers_galerkin(CFL, N, t_final)

    plt.plot(x, u, label='Numerical')
    plt.plot(x, u_exact, '--', label='Exact')
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title("Burgers' Equation via Fourier Galerkin Method")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

