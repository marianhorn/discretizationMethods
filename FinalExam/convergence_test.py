import numpy as np
import matplotlib.pyplot as plt
from RungeKutta import solve_burgers
# part 2c
def compute_error(N, t_final=np.pi/4, CFL=0.5, nu=0.1, c=4.0):
    """Run the solver and return L-infinity error at t = t_final."""
    try:
        x, u, u_exact = solve_burgers(CFL=CFL, N=N+1, t_final=t_final, nu=nu, c=c)
        error = np.max(np.abs(u - u_exact))
    except Exception as e:
        print(f"  ❌ Exception for N={N}: {type(e).__name__}: {e}")
        error = np.inf
    return error

def compute_convergence_rates(Ns, errors):
    """Compute observed convergence rates."""
    rates = []
    for i in range(1, len(Ns)):
        e1, e2 = errors[i-1], errors[i]
        N1, N2 = Ns[i-1], Ns[i]
        if e1 == 0 or e2 == 0:
            rate = np.nan
        else:
            rate = np.log(e1 / e2) / np.log(N2 / N1)
        rates.append(rate)
    return rates

def main():
    #Ns = [16, 32, 48, 64, 96, 128, 192, 256]
    Ns = [16, 32, 64, 128, 256, 512]
    t_final = np.pi / 4
    CFL = 0.01

    errors = []
    print("Measuring convergence in L∞ norm:")
    for N in Ns:
        error = compute_error(N, t_final=t_final, CFL=CFL)
        errors.append(error)
        print(f"  N={N:<3d}  Error={error:.2e}")

    # Compute convergence rates
    rates = compute_convergence_rates(Ns, errors)
    for i in range(1, len(Ns)):
        print(f"  Rate from N={Ns[i-1]} to N={Ns[i]}: {rates[i-1]:.2f}")

    # Plot error vs. N
    plt.figure(figsize=(8, 6))
    plt.loglog(Ns, errors, marker='o')
    plt.xlabel("Number of grid points N")
    plt.ylabel(r"$\|u_{\mathrm{num}} - u_{\mathrm{exact}}\|_\infty$")
    plt.title(r"Convergence of Burgers' Equation Solution at $t = \pi/4$")
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
