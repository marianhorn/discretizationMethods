import numpy as np
import matplotlib.pyplot as plt

def phi(a, b, nu, terms):
    sum_phi = np.zeros_like(a)
    for k in range(-terms, terms + 1):
        sum_phi += np.exp(-((a - (2 * k + 1) * np.pi) ** 2) / (4 * nu * b))
    return sum_phi
def dphi_dx_analytical(a, b, nu, terms):
    result = np.zeros_like(a)
    for k in range(-terms, terms + 1):
        shift = a - (2 * k + 1) * np.pi
        result += (-shift / (2 * nu * b)) * np.exp(-shift**2 / (4 * nu * b))
    return result
def analytic_solution(x, t, nu, c, terms):
    a = x - c * t
    b = t + 1
    phi_val = phi(a, b, nu, terms)
    dphi_val = dphi_dx_analytical(a, b, nu, terms)
    phi_val = np.maximum(phi_val, 1e-14)
    return c - 2 * nu * dphi_val / phi_val

def solve_burgers_fourier_galerkin(CFL, N, T, nu=0.1, c=4.0, terms=100):
    dx = 2 * np.pi / N
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    kmax = N // 2

    u0 = analytic_solution(x, 0, nu, c, terms)

    def compute_fourier_coeffs(u):
        return np.fft.fft(u) / N

    def u_from_coeffs(u_hat):
        return np.fft.ifft(u_hat * N).real

    def spectral_derivatives(u_hat):
        k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        k = 1j * k
        dudx = np.fft.ifft(k * u_hat * N).real
        d2udx2 = np.fft.ifft((k**2) * u_hat * N).real
        return dudx, d2udx2

    def F(u_hat):
        u = u_from_coeffs(u_hat)
        dudx, d2udx2 = spectral_derivatives(u_hat)
        return compute_fourier_coeffs(-u * dudx + nu * d2udx2)

    def rk4_step(u_hat, dt):
        F1 = F(u_hat)
        u1 = u_hat + 0.5 * dt * F1
        F2 = F(u1)
        u2 = u_hat + 0.5 * dt * F2
        F3 = F(u2)
        u3 = u_hat + dt * F3
        F4 = F(u3)
        return (1 / 3) * (-u_hat + u1 + 2 * u2 + u3 + 0.5 * dt * F4)

    def compute_dt(u_phys):
        umax = np.max(np.abs(u_phys))
        return CFL / (umax * kmax + nu * kmax**2)

    u_hat = compute_fourier_coeffs(u0)
    t = 0.0

    while t < T:
        u_phys = u_from_coeffs(u_hat)
        dt = compute_dt(u_phys)
        if t + dt > T:
            dt = T - t
        u_hat = rk4_step(u_hat, dt)
        t += dt

    u_num = u_from_coeffs(u_hat)
    u_ex = analytic_solution(x, T, nu, c, terms)

    return x, u_num, u_ex

def main():
    CFL = 4
    N = 128
    T = 1.0
    nu = 0.1
    c = 4.0
    terms = 100

    x, u_num, u_ex = solve_burgers_fourier_galerkin(CFL, N, T, nu=nu, c=c, terms=terms)

    plt.figure(figsize=(10, 5))
    plt.plot(x, u_num, label="Numerical (Fourier-Galerkin)", linewidth=2)
    plt.plot(x, u_ex, '--', label="Exact Solution", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.title(f"Burgers' Equation at T = {T} (CFL={CFL}, N={N})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
