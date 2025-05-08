import numpy as np
import matplotlib.pyplot as plt
from math import pi


def initial_condition(x, nu, c):
    phi = np.zeros_like(x)
    dphi = np.zeros_like(x)
    for m in range(-50, 51):
        shift = x - (2 * m + 1) * np.pi
        exp_term = np.exp(-shift**2 / (4 * nu))
        phi += exp_term
        dphi += shift * exp_term / (2 * nu)
    return c - 2 * nu * dphi / phi


def exact_solution(x, t, nu, c):
    phi = np.zeros_like(x)
    dphi = np.zeros_like(x)
    b = t + 1
    for m in range(-50, 51):
        shift = x - c * t - (2 * m + 1) * np.pi
        exp_term = np.exp(-shift**2 / (4 * nu * b))
        phi += exp_term
        dphi += shift * exp_term / (2 * nu * b)
    return c - 2 * nu * dphi / phi


def rk4_fourier_galerkin(N, CFL, T, nu, c):
    # Setup
    k = np.fft.fftfreq(N, d=1/N) * 2 * np.pi
    k = k.astype(np.complex128)
    kmax = np.max(np.abs(k))

    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    u0 = initial_condition(x, nu, c)
    u_hat = np.fft.fft(u0)

    # CFL-based time step
    u_max = np.max(np.abs(u0))
    dt = CFL / (u_max * kmax + nu * kmax**2)
    steps = int(np.round(T / dt))
    dt = T / steps

    def nonlinear_term(u_hat_in):
        u_phys = np.fft.ifft(u_hat_in).real
        u_x_hat = 1j * k * u_hat_in
        u_x = np.fft.ifft(u_x_hat).real
        nonlinear_phys = -u_phys * u_x
        return np.fft.fft(nonlinear_phys)

    def rhs(u_hat_in):
        u_xx_hat = -(k**2) * u_hat_in
        return nonlinear_term(u_hat_in) + nu * u_xx_hat

    for _ in range(steps):
        F1 = rhs(u_hat)
        u1 = u_hat + 0.5 * dt * F1
        F2 = rhs(u1)
        u2 = u_hat + 0.5 * dt * F2
        F3 = rhs(u2)
        u3 = u_hat + dt * F3
        F4 = rhs(u3)
        u_hat = (1/3) * (-u_hat + u1 + 2 * u2 + u3 + 0.5 * dt * F4)

    u_final = np.fft.ifft(u_hat).real
    return x, u_final


def main():
    N = 128
    CFL = 0.2
    T = np.pi / 4
    nu = 0.1
    c = 4.0

    x, u_num = rk4_fourier_galerkin(N, CFL, T, nu, c)
    u_exact = exact_solution(x, T, nu, c)

    error_Linf = np.max(np.abs(u_num - u_exact))
    error_L2 = np.sqrt(np.mean((u_num - u_exact)**2))

    print('--- Fourier Galerkin Method ---')
    print(f'N: {N}')
    print(f'CFL: {CFL}')
    print(f'Time: t = {T:.4f}')
    print(f'L-inf error: {error_Linf:.3e}')
    print(f'L2 error:    {error_L2:.3e}')

    plt.figure(figsize=(10, 5))
    plt.plot(x, u_num, label='Numerical')
    plt.plot(x, u_exact, '--', label='Exact')
    plt.title(f'Fourier Galerkin at t = {T:.2f}, N = {N}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
