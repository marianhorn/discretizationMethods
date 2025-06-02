import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# --- Parameters ---
theta = 0.5
sigma = 0.2
dt = 0.01
max_time = 3.0

def birth_rate(x):
    return max(0.0, 1 + 0.3 * x)

def death_rate(x):
    return 0.2


def simulate_lineage(args):
    t0, x0, lineage_id, max_id, seed = args
    np.random.seed(seed)
    t = t0
    x = x0
    trait_path = [(t, x)]
    events = []

    while t < max_time:
        dW = np.random.normal(0, np.sqrt(dt))
        x = x - theta * x * dt + sigma * dW
        t += dt
        trait_path.append((t, x))

        if np.random.rand() < birth_rate(x) * dt:
            new_id1 = max_id + 1
            new_id2 = max_id + 2
            events.append(("branch", t, lineage_id, new_id1, new_id2, x))
            return trait_path, events, [(t, x, new_id1), (t, x, new_id2)]
        if np.random.rand() < death_rate(x) * dt:
            events.append(("death", t, lineage_id))
            return trait_path, events, []

    events.append(("survive", t, lineage_id))
    return trait_path, events, []

def simulate_single_tree(seed):
    active_lineages = [(0.0, 0.0, 0)]
    max_id = 0
    all_events = []
    trait_paths = {}

    while active_lineages:
        t0, x0, lineage_id = active_lineages.pop()
        trait_path, events, new_lineages = simulate_lineage((t0, x0, lineage_id, max_id, seed + lineage_id))
        trait_paths[lineage_id] = trait_path
        all_events.extend(events)
        for t1, x1, new_id in new_lineages:
            active_lineages.append((t1, x1, new_id))
            max_id = max(max_id, new_id)

    return all_events

def matches_observed(events):
    observed_events = [("branch", 1), ("branch", 2), ("death", 2.5)]
    tolerance = 0.1
    found = {k: False for k, _ in observed_events}
    for ev in events:
        if ev[0] in ("branch", "death"):
            for (otype, otime) in observed_events:
                if ev[0] == otype and abs(ev[1] - otime) < tolerance:
                    found[otype] = True
    return all(found.values())


def parallel_likelihood(N):
    with Pool(processes=cpu_count()) as pool:
        seeds = np.random.randint(0, 1_000_000, size=N)
        results = pool.map(simulate_single_tree, seeds)
        matches = sum(matches_observed(events) for events in results)
    return matches / N

if __name__ == '__main__':
    N_vals = [10, 1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    likelihoods = [parallel_likelihood(N) for N in N_vals]

    print(f"Final Likelihood (N = {N_vals[-1]}): {likelihoods[-1]:.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(N_vals, likelihoods, marker='o')
    plt.xlabel('N (Number of Simulations)')
    plt.ylabel(r'$\hat{L}_N$')
    plt.title('Monte Carlo Estimate of Likelihood')
    plt.grid(True)
    plt.show()
