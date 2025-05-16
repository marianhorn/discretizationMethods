import numpy as np
import matplotlib.pyplot as plt
from RKgalerkin import solve_burgers_fourier_galerkin

def compute_error(N, CFL, t_final=np.pi / 4, nu=0.1, c=4.0, terms=100):
    try:
        x, u, u_exact = solve_burgers_fourier_galerkin(CFL=CFL, N=N, T=t_final, nu=nu, c=c, terms=terms)
        error = np.max(np.abs(u - u_exact))
    except Exception as e:
        print(f"Exception for N={N}: {type(e).__name__}: {e}")
        error = np.inf
    return error

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

def main():
    Ns = [16, 32, 48, 64, 96, 128, 192, 256]
    CFL_table = {
        16: 4.83,
        32: 4.42,
        48: 4.23,
        64: 4.13,
        96: 4.26,
        128: 4.30,
        192: 4.08,
        256: 3.83,
    }

    t_final = np.pi / 4
    nu = 0.1
    c = 4.0
    terms = 100

    errors = []

    print("Measuring convergence in L inf norm using individual max stable CFLs (Fourier-Galerkin):")
    for N in Ns:
        CFL = CFL_table[N]
        error = compute_error(N, CFL=CFL, t_final=t_final, nu=nu, c=c, terms=terms)
        errors.append(error)
        print(f"  N={N:<3d}  CFL={CFL:.2f}  Error={error:.2e}")

    rates = compute_convergence_rates(Ns, errors)
    for i in range(1, len(Ns)):
        print(f"  Rate from N={Ns[i-1]} to N={Ns[i]}: {rates[i-1]:.2f}")

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
