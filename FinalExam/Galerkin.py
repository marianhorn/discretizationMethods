import numpy as np
import matplotlib.pyplot as plt

# φ and its x-derivative using finite difference
def phi(a, b, nu, K=100):
    result = np.zeros_like(a)
    for k in range(-K, K + 1):
        result += np.exp(-((a - (2 * k + 1) * np.pi) ** 2) / (4 * nu * b))
    return result

def dphi_dx(a, b, nu, K=100):
    h = 1e-6
    return (phi(a + h, b, nu, K) - phi(a - h, b, nu, K)) / (2 * h)

# Exact solution to Burgers’ equation
def exact_solution(x, t, nu, c=1.0):
    a = x - c * t
    b = t + 1
    phi_val = phi(a, b, nu)
    phi_val = np.maximum(phi_val, 1e-14)  # avoid division by small values
    return c - 2 * nu * dphi_dx(a, b, nu) / phi_val

# Main solver function
def solve_burgers_fourier_galerkin(CFL, N, T, nu=0.1, c=4.0):
    dx = 2 * np.pi / (N + 1)
    x = np.linspace(0, 2 * np.pi, N + 1, endpoint=False)
    kmax = N // 2

    # Initial condition from exact solution at t = 0
    u0 = exact_solution(x, 0, nu, c)

    # FFT helpers
    def compute_fourier_coeffs(u):
        return np.fft.fft(u) / (N + 1)

    def u_from_coeffs(u_hat):
        return np.fft.ifft(u_hat * (N + 1)).real

    def spectral_derivatives(u_hat):
        k = np.fft.fftfreq(N + 1, d=dx) * 2 * np.pi
        k = 1j * k
        dudx = np.fft.ifft(k * u_hat * (N + 1)).real
        d2udx2 = np.fft.ifft((k**2) * u_hat * (N + 1)).real
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

    # Initialize
    u_hat = compute_fourier_coeffs(u0)
    t = 0.0

    # Time stepping
    while t < T:
        u_phys = u_from_coeffs(u_hat)
        dt = compute_dt(u_phys)
        if t + dt > T:
            dt = T - t
        u_hat = rk4_step(u_hat, dt)
        t += dt

    u_num = u_from_coeffs(u_hat)
    u_ex = exact_solution(x, T, nu, c)

    return x, u_num, u_ex

# Main runner
def main():
    CFL = 2
    N = 63
    T = 1.0
    x, u_num, u_ex = solve_burgers_fourier_galerkin(CFL, N, T)

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
