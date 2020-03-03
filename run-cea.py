from slgep_lib import wrap_config
from utils import Saver
from cea import cea
import argparse
import yaml


# Load configuration
config = yaml.load(open('config.yaml').read())

# Load benchmark
singletask_benchmark = yaml.load(open('atari_benchmark/singletask-benchmark.yaml').read())
instance = 'single-0'

data = singletask_benchmark[instance]
config.update(data)

seed = 0
config = wrap_config(config)
saver = Saver(config, instance, seed)

cea(config, saver.append)
