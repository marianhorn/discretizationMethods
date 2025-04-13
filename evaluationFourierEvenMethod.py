import gmpy2
from gmpy2 import mpfr, sin, cos, get_context, const_pi
import matplotlib.pyplot as plt

from fourierDiffEven import build_fourier_diff_matrix, matvec_mult

get_context().precision = 256

# Functions and their exact derivatives
functions = {
    "cos(10x)": {
        "f": lambda x: cos(10 * x),
        "df": lambda x: -10 * sin(10 * x)
    },
    "cos(x/2)": {
        "f": lambda x: cos(x / 2),
        "df": lambda x: -0.5 * sin(x / 2)
    },
    "x": {
        "f": lambda x: x,
        "df": lambda x: mpfr(1)
    }
}

def linf_error(approx, exact):
    return float(max(abs(a - e) for a, e in zip(approx, exact)))

def l2_error(approx, exact):
    return float(gmpy2.sqrt(sum((a - e) ** 2 for a, e in zip(approx, exact))))

def analyze_function(name, f_dict, N_values):
    L2_errors = []
    Linf_errors = []

    for N in N_values:
        h = 2 * const_pi() / N
        x = [i * h for i in range(N)]
        f_vals = [f_dict["f"](xi) for xi in x]
        df_exact = [f_dict["df"](xi) for xi in x]

        D = build_fourier_diff_matrix(N)
        df_approx = matvec_mult(D, f_vals)

        Linf_errors.append(linf_error(df_approx, df_exact))
        L2_errors.append(l2_error(df_approx, df_exact))

    return Linf_errors, L2_errors

def plot_errors(N_values, linf, l2, func_name):
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, linf, label="Linf (max error)", marker='o')
    plt.plot(N_values, l2, label="L2 (global error)", marker='x')
    plt.yscale("log")
    plt.xlabel("N (grid size)")
    plt.ylabel("Error (log scale)")
    plt.title(f"Fourier Differentiation Error for f(x) = {func_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"error_plot_{func_name.replace('(', '').replace(')', '').replace('/', '_')}.png")
    plt.show()

def main():
    N_values = list(range(3, 100 + 1, 2))  # Odd N

    for name, f_dict in functions.items():
        linf, l2 = analyze_function(name, f_dict, N_values)
        print(f"\n{name}:")
        for N, e_inf, e_l2 in zip(N_values, linf, l2):
            print(f"N={N:3} | Linf={e_inf:.2e} | L2={e_l2:.2e}")

        plot_errors(N_values, linf, l2, name)
if __name__ == "__main__":
    main()
