import matplotlib.pyplot as plt
import gmpy2
from gmpy2 import mpfr, sin, cos, exp, get_context, const_pi

import fourierDiffEven
import fourierDiffOdd

get_context().precision = 256

def compute_max_error(method, N, k):
    pi_val = const_pi()


    if method == 'even':
        h = 2 * pi_val / mpfr(N)
        x = [i * h for i in range(N)]
        D = fourierDiffEven.build_fourier_diff_matrix(N)
        u = [exp(k * sin(xi)) for xi in x]
        u_exact = [k * cos(xi) * exp(k * sin(xi)) for xi in x]
        u_approx = fourierDiffEven.matvec_mult(D, u)

        rel_errors = [
            abs(u_approx[i] - u_exact[i]) / abs(u_exact[i]) for i in range(N)
        ]
    elif method == 'odd':
        h = 2 * pi_val / mpfr(N+1)
        x = [i * h for i in range(N+1)]
        D = fourierDiffOdd.build_fourier_diff_matrix(N)
        u = [exp(k * sin(xi)) for xi in x]
        u_exact = [k * cos(xi) * exp(k * sin(xi)) for xi in x]
        u_approx = fourierDiffEven.matvec_mult(D, u)

        rel_errors = [
            abs(u_approx[i] - u_exact[i]) / abs(u_exact[i]) for i in range(N)
        ]
    else:
        raise ValueError("Unknown method.")

    return float(max(rel_errors))


def compare_methods(k_val=mpfr(2), Nmax=100):
    N_even = list(range(1, Nmax, 2))
    N_odd = list(range(2, Nmax+1, 2))

    even_errors = []
    odd_errors = []

    for N in N_even:
        err = compute_max_error('even', N, k_val)
        even_errors.append(err)

    for N in N_odd:
        err = compute_max_error('odd', N, k_val)
        odd_errors.append(err)

    # Plot results
    plt.figure()
    plt.plot(N_even, even_errors, label="Even method (Odd formula, Even N)", marker='o')
    plt.plot(N_odd, odd_errors, label="Odd method (Correct Fourier)", marker='s')
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("Max Relative Error")
    plt.title(f"Fourier Differentiation Error Comparison (k = {float(k_val)})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Text output
    print("\nSummary (lower error = better):\n")
    print(f"{'N':>4} | {'Even Method':>15} | {'Odd Method':>15} | {'Winner'}")
    print("-" * 55)
    for ne, no, err_e, err_o in zip(N_even, N_odd, even_errors, odd_errors):
        winner = "Odd" if err_o < err_e else "Even"
        print(f"{ne:>4} | {err_e:15.4e} | {err_o:15.4e} | {winner}")

if __name__ == "__main__":
    compare_methods(k_val=mpfr(2), Nmax=200)
