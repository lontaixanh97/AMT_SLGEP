from .operators import get_best_individual, get_statistics
from scipy.optimize import OptimizeResult


def get_optimization_results(t, population, factorial_cost, scalar_fitness, skill_factor, message):
    K = len(set(skill_factor))
    N = len(population) // 2
    results = []
    for k in range(K):
        result = OptimizeResult()
        x, fun = get_best_individual(population, factorial_cost, scalar_fitness, skill_factor, k)
        result.x = x
        result.fun = fun
        result.message = message
        result.nit = t
        result.nfev = (t + 1) * N
        mean, std = get_statistics(factorial_cost, skill_factor, k)
        result.mean = mean
        result.std = std
        results.append(result)
    return results
