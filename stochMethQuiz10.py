import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# --- Parameters ---
theta = 0.5
sigma = 0.2
dt = 0.01
max_time = 3.0
tolerance = 0.1

# Observed event pattern
observed_events = [("branch", 1), ("branch", 2), ("death", 2.5)]
required_event_count = 5  # 3 observed + 2 survives at t=3

def birth_rate(x):
    return max(0.0, 1 + 0.3 * x)

def death_rate(x):
    return 0.2

def simulate_lineage(args):
    t0, x0, lineage_id, max_id, seed = args
    np.random.seed(seed)
    t = t0
    x = x0
    events = []

    while t < max_time:
        dW = np.random.normal(0, np.sqrt(dt))
        x = x - theta * x * dt + sigma * dW
        t += dt

        if np.random.rand() < birth_rate(x) * dt:
            new_id1 = max_id + 1
            new_id2 = max_id + 2
            events.append(("branch", t))
            return events, [(t, x, new_id1), (t, x, new_id2)]

        if np.random.rand() < death_rate(x) * dt:
            events.append(("death", t))
            return events, []

    events.append(("survive", t))
    return events, []

def simulate_single_tree(seed):
    active_lineages = [(0.0, 0.0, 0)]
    max_id = 0
    all_events = []

    while active_lineages:
        t0, x0, lineage_id = active_lineages.pop()
        events, new_lineages = simulate_lineage((t0, x0, lineage_id, max_id, seed + lineage_id))
        all_events.extend(events)
        for t1, x1, new_id in new_lineages:
            active_lineages.append((t1, x1, new_id))
            max_id = max(max_id, new_id)

    return all_events

def matches_observed(events):
    for otype, otime in observed_events:
        if not any(ev[0] == otype and abs(ev[1] - otime) < tolerance for ev in events):
            return False
    if len(events) != required_event_count:
        return False
    survive_count = sum(ev[0] == "survive" and abs(ev[1] - max_time) < tolerance for ev in events)
    return survive_count == 2

def batch_simulate(num_trees, seed_base):
    matches = 0
    for i in range(num_trees):
        events = simulate_single_tree(seed_base + i)
        if matches_observed(events):
            matches += 1
    return matches

# ✅ WRAPPER FUNCTION TO FIX PICKLING ISSUE
def batch_simulate_wrapper(args):
    return batch_simulate(*args)

def parallel_likelihood_up_to_N(N_max, batch_size, checkpoints):
    seeds = np.random.randint(0, 1_000_000, size=(N_max // batch_size) * batch_size)
    batches = [(batch_size, seeds[i * batch_size]) for i in range(N_max // batch_size)]

    total_matches = 0
    total_simulated = 0
    likelihoods = []
    checkpoint_index = 0
    next_checkpoint = checkpoints[checkpoint_index]

    with Pool(processes=cpu_count()) as pool:
        for result in pool.imap_unordered(batch_simulate_wrapper, batches):
            total_matches += result
            total_simulated += batch_size
            while checkpoint_index < len(checkpoints) and total_simulated >= checkpoints[checkpoint_index]:
                likelihoods.append(total_matches / total_simulated)
                checkpoint_index += 1
                if checkpoint_index < len(checkpoints):
                    next_checkpoint = checkpoints[checkpoint_index]

            if checkpoint_index >= len(checkpoints):
                break

    return checkpoints[:len(likelihoods)], likelihoods

# --- Main Execution Block ---
if __name__ == '__main__':
    N_vals = [10, 1000] + list(range(10000, 1000001, 10000))
    checkpoints, likelihoods = parallel_likelihood_up_to_N(N_max=1000000, batch_size=1000, checkpoints=N_vals)

    print(f"Final Likelihood (N = {checkpoints[-1]}): {likelihoods[-1]:.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(checkpoints, likelihoods, marker='o')
    plt.xlabel('N (Number of Simulations)')
    plt.ylabel(r'$\hat{L}_N$')
    plt.title('Monte Carlo Estimate of Likelihood')
    plt.grid(True)
    plt.savefig("likelihood_plot.png", dpi=300)
    plt.show()
