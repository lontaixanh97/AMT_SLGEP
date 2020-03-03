import numpy as np

# MULTIFACTORIAL EVOLUTIONARY HELPER FUNCTIONS
def find_relative(population, skill_factor, sf, N):
    return population[np.random.choice(np.where(skill_factor[:N] == sf)[0])]

def calculate_scalar_fitness(factorial_cost):
    return 1 / np.min(np.argsort(np.argsort(factorial_cost, axis=0), axis=0) + 1, axis=1)

# OPTIMIZATION RESULT HELPERS
def get_best_individual(population, factorial_cost, scalar_fitness, skill_factor, sf):
    # select individuals from task sf
    idx                = np.where(skill_factor == sf)[0]
    subpop             = population[idx]
    sub_factorial_cost = factorial_cost[idx]
    sub_scalar_fitness = scalar_fitness[idx]

    # select best individual
    idx = np.argmax(sub_scalar_fitness)
    x = subpop[idx]
    fun = sub_factorial_cost[idx, sf]
    return x, fun

def get_statistics(factorial_cost, skill_factor, sf):
    idx                = np.where(skill_factor == sf)[0]
    sub_factorial_cost = factorial_cost[idx][:, sf]
    return np.mean(sub_factorial_cost), np.std(sub_factorial_cost)
