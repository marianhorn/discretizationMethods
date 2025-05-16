import numpy as np
import matplotlib.pyplot as plt
from galerkin_solver import solve_burgers_fourier_galerkin  # Make sure this path matches your file

# Compute max-norm error for given N and its max stable CFL
def compute_error(N, CFL, t_final=np.pi / 4, nu=0.1, c=4.0, terms=100):
    """Run Fourier-Galerkin solver and return L∞ error at t = t_final."""
    try:
        x, u, u_exact = solve_burgers_fourier_galerkin(CFL=CFL, N=N, T=t_final, nu=nu, c=c, terms=terms)
        error = np.max(np.abs(u - u_exact))
    except Exception as e:
        print(f"  ❌ Exception for N={N}: {type(e).__name__}: {e}")
        error = np.inf
    return error

# Compute convergence rates between resolutions
def compute_convergence_rates(Ns, errors):
    rates = []
    for i in range(1, len(Ns)):
        e1, e2 = errors[i-1], errors[i]
        N1, N2 = Ns[i-1], Ns[i]
        if e1 == 0 or e2 == 0:
            rates.append(np.nan)
        else:
            rates.append(np.log(e1 / e2) / np.log(N2 / N1))
    return rates

# Main script
def main():
    Ns = [16, 32, 48, 64, 96, 128, 192, 256]
    CFL_table = {
        16:  1.22,
        32:  1.11,
        48:  0.99,
        64:  0.91,
        96:  0.88,
        128: 0.80,
        192: 0.68,
        256: 0.60,
    }

    t_final = np.pi / 4
    nu = 0.1
    c = 4.0
    terms = 100

    errors = []

    print("Measuring convergence in L∞ norm using individual max stable CFLs (Fourier-Galerkin):")
    for N in Ns:
        CFL = CFL_table[N]
        error = compute_error(N, CFL=CFL, t_final=t_final, nu=nu, c=c, terms=terms)
        errors.append(error)
        print(f"  N={N:<3d}  CFL={CFL:.2f}  Error={error:.2e}")

    # Compute and display convergence rates
    rates = compute_convergence_rates(Ns, errors)
    for i in range(1, len(Ns)):
        print(f"  Rate from N={Ns[i-1]} to N={Ns[i]}: {rates[i-1]:.2f}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.loglog(Ns, errors, marker='o')
    plt.xlabel("Number of grid points N")
    plt.ylabel(r"$\|u_{\mathrm{num}} - u_{\mathrm{exact}}\|_\infty$")
    plt.title(r"Convergence of Fourier-Galerkin Solution at $t = \pi/4$ using Max Stable CFL")
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
