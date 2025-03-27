import numpy as np
import matplotlib.pyplot as plt


def build_fourier_diff_matrix(N):
    D_tilde = np.zeros((N + 1, N + 1), dtype=float)

    for j in range(N + 1):
        for i in range(N + 1):
            if i != j:
                sign_factor = (-1) ** (j + i)
                denom = np.sin((j - i) * np.pi / (N + 1))
                D_tilde[j, i] = sign_factor / (2.0 * denom)
            else:
                D_tilde[j, i] = 0.0
    return D_tilde


def compute_errors_for_k(k, Nmax=300):
    """
    For a given k, compute the maximum relative error for N=2..Nmax.
    Returns two arrays (N_values, max_errors):
      - N_values[i] = i-th N in [2..Nmax]
      - max_errors[i] = maximum relative error for that N
    """
    N_values = np.arange(2, Nmax + 1,2)
    max_errors = np.zeros_like(N_values, dtype=float)

    for idx, N in enumerate(N_values):
        h = 2.0 * np.pi / (N + 1)
        x = np.arange(N + 1) * h
        # Build matrix
        D_tilde = build_fourier_diff_matrix(N)

        # Evaluate function and exact derivative
        u = np.exp(k * np.sin(x))
        u_exact_deriv = k * np.cos(x) * u

        # Approx derivative
        u_approx_deriv = D_tilde @ u

        # Relative error
        denom = np.abs(u_exact_deriv) + 1e-30
        rel_err = np.abs(u_approx_deriv - u_exact_deriv) / denom
        max_rel_err = np.max(rel_err)

        max_errors[idx] = max_rel_err

        # Debug print (print every 10 steps to reduce spam):
        if (N % 10) == 0:
            print(f"[DEBUG] k={k}, N={N}: max_rel_error = {max_rel_err:.3e}")

    return N_values, max_errors


def analyze_and_plot(k_list, max_err_threshold=1e-5, Nmax=300):

    plt.figure()

    # We'll store the minimal N found for each k
    minimal_N_for_k = {}

    for k in k_list:
        # Compute the entire curve of errors
        N_vals, errs = compute_errors_for_k(k, Nmax)

        # Find first N with error < threshold
        below_thresh = np.where(errs < max_err_threshold)[0]
        if len(below_thresh) == 0:
            minN = None
        else:
            minN = N_vals[below_thresh[0]]

        minimal_N_for_k[k] = minN

        # Plot error vs. N
        plt.plot(N_vals, errs, label=f"k={k}")

    # Final formatting of the plot
    plt.xlabel("N")
    plt.ylabel("Max Relative Error")
    plt.title("Max Relative Error of Fourier Differentiation for u(x)=exp[k sin(x)]")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)


    # Print summary
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
