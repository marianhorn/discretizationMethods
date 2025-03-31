import numpy as np
import matplotlib.pyplot as plt

# Use higher precision
dtype_high = np.longdouble

def build_fourier_diff_matrix(N):
    D_tilde = np.zeros((N + 1, N + 1), dtype=dtype_high)

    for j in range(N + 1):
        for i in range(N + 1):
            if i != j:
                sign_factor = (-1) ** (j + i)
                denom = np.sin((j - i) * np.pi / (N + 1))
                D_tilde[j, i] = dtype_high(sign_factor) / (2.0 * dtype_high(denom))
            else:
                D_tilde[j, i] = dtype_high(0.0)
    return D_tilde


def compute_errors_for_k(k, Nmax=300):
    N_values = np.arange(2, Nmax + 1, 2)
    max_errors = np.zeros_like(N_values, dtype=dtype_high)

    for idx, N in enumerate(N_values):
        h = dtype_high(2.0 * np.pi / (N + 1))
        x = np.arange(N + 1, dtype=dtype_high) * h

        D_tilde = build_fourier_diff_matrix(N)

        u = np.exp(k * np.sin(x))
        u_exact_deriv = k * np.cos(x) * u
        u_approx_deriv = D_tilde @ u

        denom = np.abs(u_exact_deriv) + dtype_high(1e-30)
        rel_err = np.abs(u_approx_deriv - u_exact_deriv) / denom
        max_rel_err = np.max(rel_err)

        max_errors[idx] = max_rel_err

        if (N % 10) == 0:
            print(f"[DEBUG] k={k}, N={N}: max_rel_error = {max_rel_err:.3e}")

    return N_values, max_errors


def analyze_and_plot(k_list, max_err_threshold=1e-5, Nmax=300):
    plt.figure()
    minimal_N_for_k = {}

    for k in k_list:
        N_vals, errs = compute_errors_for_k(k, Nmax)
        below_thresh = np.where(errs < max_err_threshold)[0]
        minN = N_vals[below_thresh[0]] if len(below_thresh) > 0 else None
        minimal_N_for_k[k] = minN
        plt.plot(N_vals, errs, label=f"k={k}")

    plt.xlabel("N")
    plt.ylabel("Max Relative Error")
    plt.title("Max Relative Error of Fourier Differentiation for u(x)=exp[k sin(x)]")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)

    print("\nSummary of minimal N to achieve error < 1e-5:\n")
    for k in k_list:
        minN = minimal_N_for_k[k]
        if minN is None:
            print(f"k={k}:  No N up to {Nmax} achieved error < 1e-5.")
        else:
            print(f"k={k}:  minimal N = {minN}.")
    plt.show()


if __name__ == "__main__":
    k_list = [2, 4, 6, 8, 10, 12]
    analyze_and_plot(k_list, max_err_threshold=1e-5, Nmax=300)
