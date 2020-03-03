import numpy as np
from copy import deepcopy
from collections import namedtuple

ChromosomeRange = namedtuple('ChromosomeRange', ('R1', 'R2', 'R3', 'R4'))
# | --- Function Set --- | --- ADF Set --- | --- ADF Terminal Set --- | --- Terminals --- |
# | 'function_set'       | 'adf_set'       | 'adf_terminal_set'       | 'terminal_set'    |
# |                      |                 |        (Variables)       |     (Inputs)      |
# 0 ---------------------| R1 -------------| R2 ----------------------| R3 ---------------| R4

class Node:

    def __init__(self, index, arity, parent, chromosome_factory):
        self.index = index
        self.arity = arity
        self.parent = parent
        self.children = []
        self.chromosome_factory = chromosome_factory

    def _set_adfs_terminals(self, inputs):
        config = self.chromosome_factory.config
        for i in range(len(config['adf_terminal_set'])):
            config['adf_terminal_set'][i]['value'] = inputs[i]

    def get_value(self):
        config = self.chromosome_factory.config
        # Extract range
        R1, R2, R3, R4 = self.chromosome_factory.chromosome_range
        # If this node is a leaf, return its value
        if self.index >= R3:
            return config['terminal_set'][self.index - R3]['value']
        if self.index >= R2:
            return config['adf_terminal_set'][self.index - R2]['value']
        # If this node is a function or ADF node, 
        # we need to pass in its children as params
        params = []
        for child in self.children:
            value = child.get_value()
            if np.isnan(value):
                return float('nan')
            params.append(child.get_value())
        # If this node is an auto defined function
        # Assign input to the ADF variables
        if self.index >= R1:
            self._set_adfs_terminals(params)
            return config['adf_set'][self.index - R1]['func'].get_value()
        # If this node is a normal function
        function = config['function_set'][self.index]
        return config['function_set'][self.index]['func'](*params)

class ADF:

    def __init__(self, gene, chromosome_factory):
        self.gene = gene
        self.root = None
        self.chromosome_factory = chromosome_factory
        self._parse()

    def _parse(self):
        config = self.chromosome_factory.config
        symbols = config['function_set'] + config['adf_set'] + \
                  config['terminal_set'] + config['adf_terminal_set']

        gene = deepcopy(self.gene).tolist()

        # Assign root
        self.root = Node(index=gene[0],
                         arity=symbols[gene[0]]['arity'],
                         parent=None,
                         chromosome_factory=self.chromosome_factory)
        queue = [self.root]
        gene.pop(0)

        # Traverse BFS to build tree
        while len(queue) and len(gene):
            parent = queue.pop(0)

            for i in range(parent.arity):
                node = Node(index=gene[0],
                            arity=symbols[gene[0]]['arity'],
                            parent=parent,
                            chromosome_factory=self.chromosome_factory)
                queue.append(node)
                gene.pop(0)
                parent.children.append(node)

    def get_value(self):
        return self.root.get_value()

class ChromosomeFactory:

    def __init__(self, _config):
        self.config = _config
        # Assign defined structure of the solution
        config = _config
        # Compute chromosome range
        R1 = len(config['function_set'])
        R2 = R1 + len(config['adf_set'])
        R3 = R2 + len(config['adf_terminal_set'])
        R4 = R3 + len(config['terminal_set'])
        self.chromosome_range = ChromosomeRange(R1, R2, R3, R4)

    def _get_feasible_range(self, i):
        R1, R2, R3, R4 = self.chromosome_range
        config = self.config
        # gene at i belong to one of the given mains
        if i < config['num_main'] * (config['h_main'] + config['l_main']):
            if i % (config['h_main'] + config['l_main']) < config['h_main']:
                # Head of main: adf_set and function_set
                return 0, R2
            else:
                # Tail of main: terminal_set
                return R3, R4
        if (i - config['num_main'] * (config['h_main'] + config['l_main'])) % \
                (config['h_adf'] + config['l_adf']) < config['h_adf']:
            # Head of ADF: function_set
            return 0, R1
        else:
            # Tail of ADF: adf_terminal_set
            return R2, R3

    def initialize(self):
        config = self.config
        population = np.empty([config['pop_size'] * config['K'] * 2, config['dim']])
        for j in range(config['dim']):
            low, high = self._get_feasible_range(j)
            population[:, j] = np.random.randint(low, high, size=config['pop_size'] * config['K'] * 2)
        return population.astype(np.int32)

    def parse(self, chromosome):
        # Parse the auto defined functions
        config = self.config
        for i in range(config['num_adf']):
            head = config['num_main'] * (config['h_main'] + config['l_main']) + \
                   i * (config['h_adf'] + config['l_adf'])
            tail = head + config['h_adf'] + config['l_adf']
            config['adf_set'][i]['func'] = ADF(chromosome[head:tail], self)

        # Parse the main program
        for i in range(config['num_main']):
            head = i * (config['h_main'] + config['l_main'])
            tail = head + config['h_main'] + config['l_main']
            config['main'].append(ADF(chromosome[head:tail], self))

    def _set_main_terminals(self, inputs):
        config = self.config
        for i in range(len(config['terminal_set'])):
            config['terminal_set'][i]['value'] = inputs[i]

    def get_value(self, inputs):
        config = self.config
        self._set_main_terminals(inputs)
        outputs = []
        for i in range(config['num_main']):
            outputs.append(config['main'][i].get_value())
        return outputs

    def get_action(self, inputs):
        config = self.config
        self._set_main_terminals(inputs)
        outputs = []
        for i in range(config['num_main']):
            outputs.append(config['main'][i].get_value())
        outputs = np.array(outputs)
        outputs[np.where(outputs == np.nan)[0]] = -np.inf
        return np.argmax(outputs)

    def one_point_crossover(self, pa, pb):
        D = len(pa)
        index = np.random.randint(low=1, high=D-1)
        ca = np.empty_like(pa)
        cb = np.empty_like(pa)

        ca = np.concatenate([pa[:index], pb[index:]])
        cb = np.concatenate([pb[:index], pb[index:]])
        return ca, cb

    def _get_crossover_range(self, i):
        config = self.config
        n, h, l = config['num_main'], config['h_main'], config['l_main']
        n_adf, h_adf, l_adf = config['num_adf'], config['h_adf'], config['l_adf']
        if i < n * (h + l):
            if i % (l + h) == 0:
                low = (i / (h + l) - 1) * (h + l)
                high = (i / (h + l) + 1) * (h + l)
            else:
                low = np.floor(i / (h + l)) * (h + l)
                high = np.ceil(i / (h + l)) * (h + l)
        else:
            j = i - n * (h + l)
            if j % (l_adf + h_adf) == 0:
                low = (j / (h_adf + l_adf) - 1) * (h_adf + l_adf)
                high = (j / (h_adf + l_adf) + 1) * (h_adf + l_adf)
            else:
                low = np.floor(j / (h_adf + l_adf)) * (h_adf + l_adf)
                high = np.ceil(j / (h_adf + l_adf)) * (h_adf + l_adf)
            low += n * (h + l)
            high += n * (h + l)
        return int(low), int(high)

    def one_point_crossover_adf(self, pa, pb):
        D = len(pa)
        i = np.random.randint(low=1, high=D-1)
        low, high = self._get_crossover_range(i)
        ca = deepcopy(pa)
        cb = deepcopy(pa)
        if np.random.rand() < 0.5:
            ca[low:i] = pb[low:i]
            cb[low:i] = pa[low:i]
        else:
            ca[i:high] = pb[i:high]
            cb[i:high] = pa[i:high]
        return ca, cb

    def one_point_crossover_adf_multitask(self, pa, pb):
        D = len(pa)
        config = self.config
        low = config['num_main'] * (config['h_main'] + config['l_main'])
        i = np.random.randint(low=low + 1, high=D-1)
        low, high = self._get_crossover_range(i)
        ca = deepcopy(pa)
        cb = deepcopy(pa)
        if np.random.rand() < 0.5:
            ca[low:i] = pb[low:i]
            cb[low:i] = pa[low:i]
        else:
            ca[i:high] = pb[i:high]
            cb[i:high] = pa[i:high]
        return ca, cb

    def uniform_mutate(self, p, mutation_rate):
        c = deepcopy(p)
        for i in range(len(p)):
            if np.random.rand() < mutation_rate:
                low, high = self._get_feasible_range(i)
                c[i] = np.random.randint(low, high)
        return c

    def shorten_one_func_of_main(self, p, p_h_main):
        c = deepcopy(p)

        config = self.config
        c_h_main = p_h_main
        c_l_main = c_h_main * (config['max_arity'] - 1) + 1

        max_sum_arity = config['max_arity'] * c_h_main
        R1, R2, R3, R4 = self.chromosome_range

        sub_ind = []

        not_main_part = c[config['num_main']*(c_h_main+c_l_main):config['dim']]

        for i in range(config['num_main']):
            sum_arity = 0
            sub_tree = c[i*(c_h_main+c_l_main):(i+1)*(c_h_main+c_l_main)]

            for j in range(c_h_main):
                if sub_tree[j] < R1: sum_arity += FUNCTION_SET[sub_tree[j]]["arity"]
                else: sum_arity += config["max_arity"]

            last_arity = config["max_arity"]
            if sub_tree[c_h_main - 1] < R1: last_arity = FUNCTION_SET[sub_tree[c_h_main - 1]]["arity"]

            head_del = c_h_main - 1
            tail_del = c_h_main + c_l_main - (max_sum_arity - sum_arity) - last_arity

            sub_tree[head_del] = sub_tree[tail_del]
            sub_tree = np.delete(sub_tree, tail_del)

            sub_h_main = c_h_main - 1
            sub_l_main = sub_h_main * (config['max_arity'] - 1) + 1
            sub_tree = sub_tree[:sub_h_main+sub_l_main]

            sub_ind.extend(sub_tree)

        sub_ind.extend(not_main_part)

        return np.array(sub_ind, dtype=np.int32), sub_h_main
