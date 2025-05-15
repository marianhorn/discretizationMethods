import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 63                        # N+1 grid points
nu = 0.1                     # Viscosity
CFL = 1                     # CFL constant
T = 1.0                       # Final time
kmax = N // 2
dx = 2 * np.pi / (N + 1)
c = 4.0                       # wave speed in exact solution

# Grid
x = np.linspace(0, 2 * np.pi, N + 1, endpoint=False)

# φ function and derivative for analytical solution
def phi(a, b, K=50):
    sum_phi = np.zeros_like(a)
    for k in range(-K, K + 1):
        sum_phi += np.exp(-((a - (2 * k + 1) * np.pi)**2) / (4 * nu * b))
    return sum_phi

def dphi_dx(a, b, K=50):
    # Finite difference approximation of dφ/da
    h = 1e-6
    return (phi(a + h, b, K) - phi(a - h, b, K)) / (2 * h)

def exact_solution(x, t, c=1.0):
    a = x - c * t
    b = t + 1
    return c - 2 * nu * dphi_dx(a, b) / phi(a, b)

# Use exact solution at t=0 as initial condition
u0 = exact_solution(x, t=0, c=c)

# Fourier transform helpers
def compute_fourier_coeffs(u):
    return np.fft.fft(u) / (N + 1)

def u_from_coeffs(u_hat):
    return np.fft.ifft(u_hat * (N + 1)).real

# Derivatives in spectral space
def spectral_derivatives(u_hat):
    k = np.fft.fftfreq(N + 1, d=dx) * 2 * np.pi
    k = 1j * k
    dudx = np.fft.ifft(k * u_hat * (N + 1)).real
    d2udx2 = np.fft.ifft((k**2) * u_hat * (N + 1)).real
    return dudx, d2udx2

# RHS of Burgers' equation
def F(u_hat):
    u = u_from_coeffs(u_hat)
    dudx, d2udx2 = spectral_derivatives(u_hat)
    return compute_fourier_coeffs(-u * dudx + nu * d2udx2)

# 4th-order RK time step
def rk4_step(u_hat, dt):
    F1 = F(u_hat)
    u1 = u_hat + 0.5 * dt * F1
    F2 = F(u1)
    u2 = u_hat + 0.5 * dt * F2
    F3 = F(u2)
    u3 = u_hat + dt * F3
    F4 = F(u3)
    return (1 / 3) * (-u_hat + u1 + 2 * u2 + u3 + 0.5 * dt * F4)

# CFL-based time step
def compute_dt(u):
    umax = np.max(np.abs(u))
    return CFL / (umax * kmax + nu * kmax**2)

# Initialize and solve
u_hat = compute_fourier_coeffs(u0)
t = 0.0

while t < T:
    u_phys = u_from_coeffs(u_hat)
    dt = compute_dt(u_phys)
    if t + dt > T:
        dt = T - t
    u_hat = rk4_step(u_hat, dt)
    t += dt

# Get numerical and exact solution at final time
u_num = u_from_coeffs(u_hat)
u_ex = exact_solution(x, T, c=c)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, u_num, label="Numerical (Fourier-Galerkin)", linewidth=2)
plt.plot(x, u_ex, '--', label="Exact solution", linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x, T)")
plt.title(f"Burgers' Equation at t = {T} (Initial condition from exact solution)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
