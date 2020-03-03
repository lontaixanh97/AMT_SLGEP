from .data_structure import *

def wrap_config(config):
    config['l_main']       = config['h_main'] * (config['max_arity'] - 1) + 1
    config['l_adf']        = config['h_adf'] * (config['max_arity'] - 1) + 1
    config['dim']          = config['num_main'] * (config['h_main'] + config['l_main']) + \
                             config['num_adf'] * (config['h_adf'] + config['l_adf'])
    config['function_set'] = create_function_set()
    config['adf_set']      = create_adfs_set(config)
    config['adf_terminal_set'] = create_adfs_terminal_set(config)
    config['terminal_set'] = create_terminal_set(config)
    config['main']         = []
    config['mutation_rate'] = (config['h_adf'] + config['l_adf']) / config['dim']
    return config

