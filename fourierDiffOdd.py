import gmpy2
from gmpy2 import mpfr, sin, cos, exp, get_context
import matplotlib.pyplot as plt

# To run the simulation for install the packages by pip install gmpy2 matplotlib

# Set precision (in bits)
get_context().precision = 256

def build_fourier_diff_matrix(N):
    D_tilde = [[mpfr(0) for _ in range(N + 1)] for _ in range(N + 1)]
    pi = gmpy2.const_pi()

    for j in range(N + 1):
        for i in range(N + 1):
            if i != j:
                sign = (-1) ** (j + i)
                angle = (j - i) * pi / mpfr(N + 1)
                D_tilde[j][i] = mpfr(sign) / (2 * sin(angle))
    return D_tilde

def matvec_mult(D, u):
    N = len(u)
    result = [mpfr(0) for _ in range(N)]
    for i in range(N):
        for j in range(N):
            result[i] += D[i][j] * u[j]
    return result

def compute_errors_for_k(k_val, Nmax=300):
    N_values = list(range(2, Nmax + 1, 2))  # Only even N
    max_errors = []

    for N in N_values:
        h = 2 * gmpy2.const_pi() / mpfr(N + 1)
        x = [i * h for i in range(N + 1)]

        D = build_fourier_diff_matrix(N)

        u = [exp(k_val * sin(xi)) for xi in x]
        u_exact = [k_val * cos(xi) * exp(k_val * sin(xi)) for xi in x]

        u_approx = matvec_mult(D, u)

        rel_errors = [
            abs(u_approx[i] - u_exact[i]) / (abs(u_exact[i]))
            for i in range(N + 1)
        ]
        max_err = max(rel_errors)
        max_errors.append(max_err)

        if N % 10 == 0:
            print(f"[DEBUG] k={float(k_val)}, N={N}: max_rel_error = {float(max_err):.3e}")

    return N_values, max_errors

def analyze_and_plot(k_list, max_err_threshold=1e-5, Nmax=300):
    plt.figure()
    minimal_N_for_k = {}

    for k_float in k_list:
        k = mpfr(k_float)
        N_vals, errs_mp = compute_errors_for_k(k, Nmax)

        errs = [float(e) for e in errs_mp]

        minN = next((N for N, err in zip(N_vals, errs) if err < max_err_threshold), None)
        minimal_N_for_k[k_float] = minN

        plt.plot(N_vals, errs, label=f"k={k_float}")

    plt.xlabel("N")
    plt.ylabel("Max Relative Error")
    plt.title("Fourier Differentiation Error")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    print("\nSummary of minimal N to achieve error < {:.0e}:\n".format(max_err_threshold))
    for k in k_list:
        minN = minimal_N_for_k[k]
        if minN is None:
            print(f"k={k}: No N up to {Nmax} achieved error < {max_err_threshold:.0e}.")
        else:
            print(f"k={k}: minimal N = {minN}")
    plt.show()


if __name__ == "__main__":
    k_list = [2, 4, 6, 8, 10, 12]
    analyze_and_plot(k_list, max_err_threshold=1e-5, Nmax=300)
