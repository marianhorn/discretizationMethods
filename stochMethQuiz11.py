import numpy as np
import pandas as pd
from scipy.optimize import minimize


# === Likelihood function ===
def log_likelihood(params, branching_times, n_extant):
    beta0, betaN, mu = params
    branching_times = np.sort(branching_times)
    n = 1  # number of lineages starts at 1
    logL = 0.0

    for i in range(len(branching_times)):
        t = branching_times[i]
        lambda_t = beta0 + betaN * n
        if lambda_t <= 0 or mu < 0:
            return -np.inf
        rate = lambda_t + mu
        dt = branching_times[i] - branching_times[i - 1] if i > 0 else t
        logL += np.log(lambda_t) - rate * dt
        n += 1

    # Survival term: probability of extant lineages surviving until present
    lambda_T = beta0 + betaN * n
    rate_T = lambda_T + mu
    time_since_last = 2025 - branching_times[-1]
    logL -= n_extant * rate_T * time_since_last

    return logL


# === MLE wrapper ===
def fit_ddd_model(branching_times, n_extant, initial_guess=(0.1, -0.001, 0.01)):
    neg_log_likelihood = lambda params: -log_likelihood(params, branching_times, n_extant)
    result = minimize(neg_log_likelihood, initial_guess, bounds=[(1e-5, None), (None, None), (1e-5, None)])

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    # Approximate standard errors using inverse Hessian (if available)
    try:
        hessian_inv = result.hess_inv.todense() if hasattr(result.hess_inv, "todense") else result.hess_inv
        se = np.sqrt(np.diag(hessian_inv))
    except:
        se = [np.nan] * 3

    return {
        "beta0": result.x[0],
        "betaN": result.x[1],
        "mu": result.x[2],
        "logL": -result.fun,
        "se": se
    }


# === Run the analysis ===
def main(filename="automobile_branching_times.csv", missing_tips=0):
    data = pd.read_csv(filename)
    branching_times = data['branching_time'].values

    total_models = 313 + missing_tips
    extinct_models = 313 - 96
    n_extant = total_models - extinct_models

    results = fit_ddd_model(branching_times, n_extant)

    print("=== MLE Results (missing tips: {}) ===".format(missing_tips))
    print(f"β₀  = {results['beta0']:.6f} ± {results['se'][0]:.6f}")
    print(f"βₙ  = {results['betaN']:.6f} ± {results['se'][1]:.6f}")
    print(f"μ₀  = {results['mu']:.6f} ± {results['se'][2]:.6f}")
    print(f"Log-likelihood = {results['logL']:.2f}")


if __name__ == "__main__":
    print(">> Without missing tips:")
    main("automobile_branching_times.csv", missing_tips=0)
    print("\n>> With 2 missing tips:")
    main("automobile_branching_times.csv", missing_tips=2)
