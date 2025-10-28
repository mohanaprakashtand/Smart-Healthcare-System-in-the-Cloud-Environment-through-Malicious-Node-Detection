import numpy as np
import time


# Improved Seagull Optimization Algorithm
# Proposed update is done at line 26
def PROPOSED(pop, fitness_func, lb, ub, max_iter):
    pop_size, dim = pop.sape
    fitness = np.array([fitness_func(pop[ind]) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_f = fitness[best_idx]
    Convergence = np.zeros((max_iter))
    hist = {'best_f': [], 'best_params': []}
    ct = time.time()
    for t in range(max_iter):
        T = max_iter
        # parameter that decreases with time (used to balance exploration/exploitation)
        A = 2 * (1 - (t / T))  # common in many swarm methods (kept simple)

        for i in range(pop_size):
            X_i = pop[i].copy()
            # r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            # Proposed update
            r1 = (fitness[i] + np.mean(fitness)) / (fitness[i] + np.max(fitness) + np.min(fitness))

            # ------- Migration (global exploration) -------
            # Move towards best with randomness and time-decay
            # eq: X_new = X + rand*(best - X)*A   (A reduces over time)
            X_migrate = X_i + r1 * (best - X_i) * A

            # ------- Attack (spiral local exploitation) -------
            # Spiral motion around best solution: (inspired by seagull 'attack')
            # distance vector
            dist = np.linalg.norm(best - X_i) + 1e-9
            # parameter b controls spiral tightness
            b = 1.0
            l = np.random.uniform(-1, 1, size=dim)
            # spiral update (vectorized): best + dist * exp(b*l) * cos(2*pi*l)
            X_spiral = best + (best - X_i) * np.exp(b * l) * np.cos(2 * np.pi * l)

            # choose blended update with probability depending on A
            if np.random.rand() < 0.5:
                X_new = X_migrate
            else:
                X_new = X_spiral

            # boundary handling
            X_new = np.clip(X_new, lb[i], ub[i])

            # evaluate
            f_new = fitness_func(X_new)

            # greedy selection
            if f_new < fitness[i]:
                pop[i] = X_new
                fitness[i] = f_new

                # update global best
                if f_new < best_f:
                    best_f = f_new
                    best = X_new.copy()
        hist['best_f'].append(best_f)
        hist['best_params'].append(best.copy())

        Convergence[t] = best_f
    ct = time.time() - ct
    return best_f, Convergence, best, ct
