import mpmath as mp
import matplotlib.pyplot as plt

# Set high precision (e.g., 100 decimal digits)
mp.dps = 100000

def build_fourier_diff_matrix(N):
    D_tilde = mp.matrix(N + 1, N + 1)
    pi = mp.pi

    for j in range(N + 1):
        for i in range(N + 1):
            if i != j:
                sign_factor = (-1) ** (j + i)
                angle = (j - i) * pi / (N + 1)
                denom = mp.sin(angle)
                D_tilde[j, i] = mp.mpf(sign_factor) / (2 * denom)
            else:
                D_tilde[j, i] = mp.mpf(0)
    return D_tilde


def compute_errors_for_k(k, Nmax=200):
    N_values = list(range(2, Nmax + 1, 2))  # Only odd N ≥ 3
    max_errors = []

    for N in N_values:
        h = 2 * mp.pi / (N + 1)
        x = [i * h for i in range(N + 1)]

        D_tilde = build_fourier_diff_matrix(N)

        # Function and exact derivative
        u = mp.matrix([mp.exp(k * mp.sin(xi)) for xi in x])
        u_exact_deriv = mp.matrix([k * mp.cos(xi) * mp.exp(k * mp.sin(xi)) for xi in x])

        u_approx_deriv = D_tilde * u

        # Relative error (prevent division by 0)
        rel_errors = [
            mp.fabs(u_approx_deriv[i] - u_exact_deriv[i]) / (mp.fabs(u_exact_deriv[i]) + mp.mpf('1e-80'))
            for i in range(N + 1)
        ]
        max_rel_err = max(rel_errors)
        max_errors.append(max_rel_err)

        # Print debug info

        print(f"[DEBUG] k={k}, N={N}: max_rel_error = {float(max_rel_err):.3e}")

    return N_values, max_errors


def analyze_and_plot(k_list, max_err_threshold=1e-5, Nmax=200):
    plt.figure()
    minimal_N_for_k = {}

    for k_val in k_list:
        k = mp.mpf(k_val)
        N_vals, errs_mp = compute_errors_for_k(k, Nmax)

        # Convert high-precision results to float for plotting
        errs = [float(e) for e in errs_mp]

        # Find minimal N with error < threshold
        minN = next((N for N, err in zip(N_vals, errs) if err < max_err_threshold), None)
        minimal_N_for_k[k_val] = minN

        # Plot
        plt.plot(N_vals, errs, label=f"k={k_val}")

    plt.xlabel("N")
    plt.ylabel("Max Relative Error")
    plt.title("Max Relative Error of Fourier Differentiation for u(x)=exp[k sin(x)]")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    print("\nSummary of minimal N to achieve error < {:.0e}:\n".format(max_err_threshold))
    for k_val in k_list:
        minN = minimal_N_for_k[k_val]
        if minN is None:
            print(f"k={k_val}: No N up to {Nmax} achieved error < {max_err_threshold:.0e}.")
        else:
            print(f"k={k_val}: minimal N = {minN}")
    plt.show()


if __name__ == "__main__":
    k_list = [2,4,6,8,10,12]
    analyze_and_plot(k_list, max_err_threshold=1e-5, Nmax=200)
