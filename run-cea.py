import os
from slgep_lib import wrap_config
from utils import Saver
from cea import cea
import argparse
import yaml
from tools import Tools

all_models = []
Tools.save_to_file(os.path.join("problems", 'all_models'), all_models)

# Load configuration
config = yaml.load(open('config.yaml').read())

# Load benchmark
singletask_benchmark = yaml.load(open('atari_benchmark/singletask-benchmark.yaml').read())

seeds = [0]
instances = ['single-3', 'single-4']

for seed in seeds:
    for instance in instances:
        data = singletask_benchmark[instance]
        config.update(data)
        config = wrap_config(config)
        saver = Saver(config, instance, seed)

        cea(config, saver.append, addr="problems")
        saver.save()
