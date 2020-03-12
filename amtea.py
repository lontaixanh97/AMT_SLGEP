import os
import numpy as np

from probability_models.probability_model import ProbabilityModel
from probability_models.mixture_model import MixtureModel
import os
from atari_benchmark import *
from slgep_lib import *
from mfea_lib import *
from utils import Evaluator
from tools import Tools

from joblib import Parallel, delayed
from os import cpu_count
from tqdm import trange


def AMT_BGA(config, reps, trans, addr="problems"):
    """[bestSol, fitness_hist, alpha] = TSBGA(problem, dims, reps, trans): Adaptive
        Model-based Transfer Binary GA. The crossover and mutation for this simple
        binary GA are uniform crossover and bit-flip mutation.
        INPUT:
         problem: problem type, 'onemax', 'onemin', or 'trap5'
         dims: problem dimensionality
         reps: number of repeated trial runs
         trans:    trans.transfer: binary variable
                   trans.TrInt: transfer interval for AMT

        OUTPUT:
         bestSol: best solution for each repetiion
         fitness: history of best fitness for each generation
         alpha: transfer coefficient
    """
    transfer = trans['transfer']
    if transfer:
        TrInt = trans['TrInt']
        all_models = Tools.load_from_file(os.path.join(addr, 'all_models'))

    # Problem
    taskset = Taskset(config)

    # Model
    cf = ChromosomeFactory(config)

    # Simple parameter
    K = taskset.K
    config['K'] = K
    N = config['pop_size'] * K
    T = config['num_iter']
    mutation_rate = config['mutation_rate']

    # Initialization
    population = cf.initialize()
    skill_factor = np.array([i % K for i in range(2 * N)])
    factorial_cost = np.full([2 * N, K], np.inf)
    scalar_fitness = np.empty([2 * N])

    # For parallel evaluation
    print('[+] Initializing evaluators')
    evaluators = [Evaluator(config) for _ in range(2 * N)]

    # First evaluation (sequential)
    delayed_functions = []
    for i in range(2 * N):
        sf = skill_factor[i]
        delayed_functions.append(delayed(evaluators[i].evaluate)(population[i], sf))
    fitnesses = Parallel(n_jobs=cpu_count())(delayed_functions)
    for i in range(2 * N):
        sf = skill_factor[i]
        factorial_cost[i, sf] = fitnesses[i]
    scalar_fitness = calculate_scalar_fitness(factorial_cost)



    for rep in range(reps):
        alpha_rep = []
        population = np.round(np.random.rand(pop, dims))
        if fitness_func == 'toy_problem':
            fitness = fitness_eval(population, problem, dims)
        else:
            fitness = knapsack_fitness_eval(population, problem, dims, pop)
        ind = np.argmax(fitness)
        best_fit = fitness[ind]
        print('Generation 0 best fitness = ', str(best_fit))
        fitness_hist[rep, 0] = best_fit

        for i in range(1, gen):
            # As we consider all the population as parents, we don't samplt P^{s}
            if transfer and i % TrInt == 0:
                mmodel = MixtureModel(all_models)
                mmodel.createtable(population, True, 'umd')
                mmodel.EMstacking()  # Recombination of probability models
                mmodel.mutate()  # Mutation of stacked probability model
                offspring = mmodel.sample(pop)
                alpha_rep.append(mmodel.alpha)
                print('Transfer coefficient at generation ', str(i), ': ', str(mmodel.alpha))

            else:
                parent1 = population[np.random.permutation(pop), :]
                parent2 = population[np.random.permutation(pop), :]
                tmp = np.random.rand(pop, dims)
                offspring = np.zeros((pop, dims))
                index = tmp >= 0.5
                offspring[index] = parent1[index]
                index = tmp < 0.5
                offspring[index] = parent2[index]
                tmp = np.random.rand(pop, dims)
                index = tmp < (1 / dims)
                offspring[index] = np.abs(1 - offspring[index])

            if fitness_func == 'toy_problem':
                cfitness = fitness_eval(population, problem, dims)
            else:
                cfitness = knapsack_fitness_eval(population, problem, dims, pop)
            interpop = np.append(population, offspring, 0)
            interfitness = np.append(fitness, cfitness)
            index = np.argsort((-interfitness))
            interfitness = interfitness[index]
            fitness = interfitness[:pop]
            interpop = interpop[index, :]
            population = interpop[:pop, :]
            print('Generation ', str(i), ' best fitness = ', str(np.max(fitness_hist)))
            fitness_hist[rep, i] = fitness[0]

        alpha[rep] = alpha_rep
        bestSol[rep, :] = population[ind, :]
    return bestSol, fitness_hist, alpha
