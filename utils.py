from atari_benchmark import *
from copy import deepcopy
from slgep_lib import *
import numpy as np
import pickle
import yaml
import os


class Evaluator:

    def __init__(self, config):
        self.taskset = Taskset(config)
        self.cf = []
        for h_main in config['h_mains']:
            config['h_main'] = h_main
            config['h_main_multitask'] = np.max(config['h_mains'])
            self.cf.append(ChromosomeFactory(config))

    def decode(self, chromosome, sf):
        config = self.cf[sf].config
        current_h_main = config['h_main_multitask']
        while config['h_main'] < current_h_main:
            chromosome, _ = self.cf[sf].shorten_one_func_of_main(chromosome, current_h_main)
            current_h_main -= 1
        return chromosome

    def evaluate(self, chromosome, sf):
        chromosome = self.decode(chromosome, sf)
        self.cf[sf].parse(chromosome)
        try:
            fitness = self.taskset.run_episode(sf, self.cf[sf].get_action)
        except (OverflowError, ValueError) as e:
            fitness = np.inf
        return fitness


class Saver:

    def __init__(self, config, instance, seed):
        '''Folder result/instance
                            config.yaml
                            <seed>.pkl
        Parameters
        ----------
            config (dict): configuration of the problem
            instance (str): name of the benchmark
        '''
        self.seed = seed
        self.instance = instance
        # Create result folder
        folder = 'result'
        if not os.path.exists(folder):
            os.mkdir(folder)
        folder = 'result/%s' % instance
        if not os.path.exists(folder):
            os.mkdir(folder)
        # Save configuration
        path = os.path.join(folder, 'config.yaml')
        _config = deepcopy(config)
        del _config['function_set']
        del _config['adf_set']
        del _config['adf_terminal_set']
        del _config['terminal_set']
        with open(path, 'w') as fp:
            yaml.dump(_config, fp)
        self.results = []

    def append(self, result):
        self.results.append(result)
        self.save()

    def save(self):
        path = os.path.join('result', self.instance, '%d.pkl' % self.seed)
        with open(path, 'wb') as fp:
            pickle.dump(self.results, fp, protocol=pickle.HIGHEST_PROTOCOL)
