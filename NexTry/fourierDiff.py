import numpy as np
import mpmath
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


def compare_fourier_diff_methods():
    k_vals = list(range(2, 14, 2))  # k = 2, 4, ..., 12
    precision_digits = 50
    tol = mpmath.mpf("1e-5")
    maxN = 256

    methods = ['even', 'odd']
    styles = ['-', '--']
    results = {}

    plt.figure()
    k_colors = plt.colormaps.get_cmap('tab10')

    for m_idx, method in enumerate(methods):
        method_label = method.upper()
        print(f"\n--- Running method: {method_label} ---")
        res = run_fourier_diff_analysis(k_vals, precision_digits, tol, method, maxN)

        for idx, k in enumerate(k_vals):
            k_str = str(k)
            color = k_colors(idx % 10)  # Reuse colors if more than 10
            plt.semilogy(res['N_vals'][k_str], res['max_errors'][k_str],
                         styles[m_idx], color=color,
                         label=f'{method_label}, k = {k}')

        results[method] = res

    plt.xlabel('N')
    plt.ylabel('Max Relative Error')
    plt.title('Fourier Spectral Differentiation: Even vs Odd Methods')
    plt.grid(True)
    plt.yscale('log')
    plt.ylim(1e-55, 1e10)
    plt.legend(loc='lower left')
    plt.show()


    print(f"\n=== Summary of Minimal N for Error < {float(tol):.0e} ===")
    print(f"{'k':>6} | {'Even N':>10} | {'Odd N':>10}")
    print('-' * 30)
    for k in k_vals:
        ke = results['even']['minimal_N'].get(str(k), None)
        ko = results['odd']['minimal_N'].get(str(k), None)
        print(f"{k:6d} | {printN(ke):>10} | {printN(ko):>10}")


def run_fourier_diff_analysis(k_vals, precision_digits, tol, method, maxN):
    mpmath.mp.dps = precision_digits
    minimal_N = {}
    all_errors = {}
    all_N_vals = {}

    for k in k_vals:
        N_range = list(range(2, maxN + 1, 2))
        args_list = [(N, k, precision_digits, method) for N in N_range]

        with Pool(processes=cpu_count()) as pool:
            max_errors = pool.map(evaluate_error_for_N, args_list)

        min_N = next((N for N, err in zip(N_range, max_errors) if err < float(tol)), None)
        k_str = str(k)
        minimal_N[k_str] = min_N
        all_errors[k_str] = max_errors
        all_N_vals[k_str] = N_range

    return {
        "minimal_N": minimal_N,
        "max_errors": all_errors,
        "N_vals": all_N_vals
    }


def evaluate_error_for_N(args):
    N, k, precision_digits, method = args
    mpmath.mp.dps = precision_digits

    D, x = fourier_diff_matrix_vpa(N, method)
    u = [mpmath.exp(k * mpmath.sin(xi)) for xi in x]
    du_true = [k * mpmath.cos(xi) * mpmath.exp(k * mpmath.sin(xi)) for xi in x]
    du_num = [sum(D[i][j] * u[j] for j in range(len(u))) for i in range(len(u))]

    rel_error = [abs((du_n - du_t) / du_t) for du_n, du_t in zip(du_num, du_true)]
    max_rel_err = max(rel_error)
    return float(max_rel_err)


def fourier_diff_matrix_vpa(N, method):
    if method == 'even':
        x = [mpmath.mpf(2) * mpmath.pi * i / N for i in range(N)]
        D = [[mpmath.mpf(0) for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i][j] = mpmath.mpf(0.5) * (-1)**(i + j) / mpmath.tan((x[i] - x[j]) / 2)
    elif method == 'odd':
        Nsym = N + 1
        h = mpmath.mpf(2) * mpmath.pi / Nsym
        x = [h * i for i in range(Nsym)]
        D = [[mpmath.mpf(0) for _ in range(Nsym)] for _ in range(Nsym)]
        for j in range(Nsym):
            for i in range(Nsym):
                if i != j:
                    D[j][i] = (-1)**(i + j) / (2 * mpmath.sin((j - i) * mpmath.pi / Nsym))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'even' or 'odd'.")
    return D, x


def printN(N):
    return str(N) if N is not None else "—"


if __name__ == "__main__":
    compare_fourier_diff_methods()
