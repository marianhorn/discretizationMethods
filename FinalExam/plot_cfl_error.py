import numpy as np
import matplotlib.pyplot as plt
from RungeKutta import solve_burgers
# PART 2 b
def count_error(CFL, N, t_final=0.5, nu=0.1, c=4.0):
    """Run solver and return max-norm error for given CFL and N."""
    try:
        x, u, u_exact = solve_burgers(CFL=CFL, N=N+1, t_final=t_final, nu=nu, c=c)
        error = np.max(np.abs(u - u_exact))
    except Exception as e:
        print(f"   Exception for CFL={CFL:.2f}, N={N}: {type(e).__name__}: {e}")
        error = np.inf
    return error

def main():
    Ns = [16, 32, 64, 128, 256]
    CFL_values = np.linspace(0.1, 2.0, 20)
    t_final = 0.5

    plt.figure(figsize=(8, 6))

    for N in Ns:
        errors = []
        print(f"Running CFL sweep for N = {N}")
        for CFL in CFL_values:
            error = count_error(CFL, N, t_final=t_final)
            errors.append(error)
            print(f"  CFL={CFL:.2f}  Error={error:.2e}")
        plt.plot(CFL_values, errors, label=f'N = {N}', marker='o')

    plt.xlabel("CFL number")
    plt.ylabel(r"$\|u_{\mathrm{numerical}} - u_{\mathrm{exact}}\|_\infty$")
    plt.title("Error vs. CFL for Various Grid Sizes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
