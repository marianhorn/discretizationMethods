import mpmath
import matplotlib.pyplot as plt


def analyze_fourier_convergence():
    mpmath.mp.dps = 50  # digits of precision

    precision_digits = 50
    method = 'even'
    N_vals = list(range(8, 65, 4))  # N = 8, 12, ..., 64

    funcs = [
        lambda x: [mpmath.cos(10 * xi) for xi in x],
        lambda x: [mpmath.cos(xi / 2) for xi in x],
        lambda x: list(x)
    ]
    dfuncs = [
        lambda x: [-10 * mpmath.sin(10 * xi) for xi in x],
        lambda x: [-0.5 * mpmath.sin(xi / 2) for xi in x],
        lambda x: [mpmath.mpf(1) for _ in x]
    ]
    labels = ['cos(10x)', 'cos(x/2)', 'x']

    for f_idx, (f, df, label) in enumerate(zip(funcs, dfuncs, labels)):
        print(f"\n=== Processing function: {label} ===")

        Linf_errors = []
        L2_errors = []

        for N in N_vals:
            print(f"[DEBUG] f = {label}, N = {N} → building matrix...")
            D, x = fourier_diff_matrix_vpa(N, precision_digits, method)

            fx = f(x)
            dfx_true = df(x)
            dfx_approx = [sum(D[i][j] * fx[j] for j in range(len(fx))) for i in range(len(fx))]

            err = [abs(dfx_true[i] - dfx_approx[i]) for i in range(len(fx))]
            Linf = max(err)
            L2 = mpmath.sqrt(sum(e**2 for e in err) / len(err))

            Linf_errors.append(Linf)
            L2_errors.append(L2)

            print(f"[DEBUG]    Linf error = {float(Linf):.3e} | L2 error = {float(L2):.3e}")

        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Error convergence: {label}', fontsize=14)

        axs[0].semilogy(N_vals, [float(e) for e in L2_errors], '-o', linewidth=1.5)
        axs[0].set_title(f'L2 Error – {label}')
        axs[0].set_xlabel('N')
        axs[0].set_ylabel('L2 error')
        axs[0].grid(True)

        axs[1].semilogy(N_vals, [float(e) for e in Linf_errors], '-s', linewidth=1.5)
        axs[1].set_title(f'L∞ Error – {label}')
        axs[1].set_xlabel('N')
        axs[1].set_ylabel('L∞ error')
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

        # Summary Table
        print(f"\n{'Function':<12} | {'N':<10} | {'L∞ error':<15} | {'L2 error':<15}")
        print('-' * 60)
        for N, Linf, L2 in zip(N_vals, Linf_errors, L2_errors):
            print(f"{label:<12} | {N:<10} | {mpmath.nstr(Linf, 5):<15} | {mpmath.nstr(L2, 5):<15}")
        print('-' * 60)


def fourier_diff_matrix_vpa(N, precision_digits, method):
    mpmath.mp.dps = precision_digits

    if method == 'even':
        x = [2 * mpmath.pi * i / N for i in range(N)]
        D = [[mpmath.mpf(0) for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i][j] = 0.5 * (-1)**(i + j) / mpmath.tan((x[i] - x[j]) / 2)
    else:
        raise ValueError("Only 'even' method is supported in this script.")
    return D, x


if __name__ == "__main__":
    analyze_fourier_convergence()
