import numpy as np
import matplotlib.pyplot as plt
from RKgalerkin import solve_burgers_fourier_galerkin  # <- import your Fourier-Galerkin solver

def count_error(CFL, N, t_final=0.5, nu=0.1, c=4.0, terms=100):
    """Run Fourier-Galerkin solver and return max-norm error for given CFL and N."""
    try:
        x, u, u_exact = solve_burgers_fourier_galerkin(CFL=CFL, N=N, T=t_final, nu=nu, c=c, terms=terms)
        error = np.max(np.abs(u - u_exact))
    except Exception as e:
        print(f"   Exception for CFL={CFL:.2f}, N={N}: {type(e).__name__}: {e}")
        error = np.inf
    return error

def detect_max_stable_cfl(CFL_values, errors, jump_threshold=1):
    stable_index = 0
    for i in range(1, len(errors)):
        e0, e1 = errors[i-1], errors[i]
        if not (np.isfinite(e0) and np.isfinite(e1)):
            break
        rel_change = abs(e1 - e0) / max(e0, 1e-12)
        if rel_change > jump_threshold:
            break
        stable_index = i
    return CFL_values[stable_index]

def main():
    Ns = [16, 32, 48, 64, 96, 128, 192, 256]
    CFL_values = np.linspace(0.1, 5.5, 50)
    t_final = np.pi
    nu = 0.1
    c = 4.0
    terms = 100

    plt.figure(figsize=(8, 6))

    for N in Ns:
        errors = []
        print(f"Running CFL sweep for N = {N}")
        for CFL in CFL_values:
            error = count_error(CFL, N, t_final=t_final, nu=nu, c=c, terms=terms)
            errors.append(error)
            print(f"  CFL={CFL:.2f}  Error={error:.2e}")

        # Filter points with finite error
        CFL_filtered = [cfl for cfl, err in zip(CFL_values, errors) if err < 5]
        errors_filtered = [err for err in errors if err < 5]

        plt.plot(CFL_filtered, errors_filtered, label=f'N = {N}', marker='o')

    plt.xlabel("CFL number")
    plt.ylabel(r"$\|u_{\mathrm{numerical}} - u_{\mathrm{exact}}\|_\infty$")
    plt.title("Error vs. CFL for Fourier-Galerkin RK4 Method")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
