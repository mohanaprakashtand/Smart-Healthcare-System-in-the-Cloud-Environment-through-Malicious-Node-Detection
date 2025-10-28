import time
import numpy as np


def update_wave_position(current_position, velocity, bounds):
    new_position = current_position + velocity
    # Ensure the new position is within the search space bounds
    new_position = np.clip(new_position, bounds[0], bounds[1])
    return new_position


# Water Wave Optimization (WWO)
def WWO(population, fobj, VRmin, VRmax, iterations):
    pop_size, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    velocity = np.zeros_like(population)
    fitness = fobj(population)
    best_solution = np.zeros((dim, 1))
    best_fitness = float('inf')

    Convergence_curve = np.zeros((iterations, 1))

    t = 0
    ct = time.time()
    for iteration in range(iterations):
        # Update velocities
        alpha = 0.5
        beta = 0.1
        bounds = (-5, 5)
        delta = np.random.rand(pop_size, dim)
        distance = np.abs(population - population[delta.argsort(axis=1)[:, 0]])

        velocity = alpha * velocity + beta * distance * (population[delta.argsort(axis=1)[:, 1]] - population)

        # Update positions
        population = update_wave_position(population, velocity, bounds)

        # Evaluate fitness
        new_fitness = fobj(population)

        # Update population based on fitness
        better_indices = new_fitness < fitness
        population[better_indices] = population[better_indices]
        fitness[better_indices] = new_fitness[better_indices]
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[iterations - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_solution, ct
