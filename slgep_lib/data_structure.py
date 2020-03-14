import numpy as np

np.seterr(all='raise')


def add(a, b):
    try:
        c = np.add(a, b)
    #    except: print('add error', a, b); raise
    except:
        return np.nan
    return c


def subtract(a, b):
    try:
        c = np.subtract(a, b)
    #    except: print('subtract error', a, b); raise
    except:
        return np.nan
    return c


def multiply(a, b):
    try:
        c = np.multiply(a, b)
    #    except: print('multiply error', a, b); raise
    except:
        return np.nan
    return c


def divide(a, b):
    if b == 0:
        return np.nan
    try:
        c = np.divide(a, b)
    #    except: print('divide error', a, b); raise
    except:
        return np.nan
    return c


def log(a):
    if a == 0:
        return np.nan
    try:
        b = np.log(a)
    #    except: print('log error', a); raise
    except:
        return np.nan
    return b


def exp(a):
    if a >= 88:
        return np.nan
    try:
        b = np.exp(a)
    #    except: print('exp error', type(a), a); raise
    except:
        return np.nan
    return b


FUNCTION_SET = [
    {'name': '+', 'func': lambda a, b: add(a, b), 'arity': 2},
    {'name': '-', 'func': lambda a, b: subtract(a, b), 'arity': 2},
    {'name': '*', 'func': lambda a, b: multiply(a, b), 'arity': 2},
    {'name': '/', 'func': lambda a, b: divide(a, b), 'arity': 2},
    # {'name': 'sin', 'func': lambda a: np.sin(a), 'arity': 1},
    # {'name': 'cos', 'func': lambda a: np.cos(a), 'arity': 1},
    {'name': 'e^x', 'func': lambda a: exp(a), 'arity': 1},
    {'name': 'ln|x|', 'func': lambda a: log(np.abs(a)), 'arity': 1}
]


def create_adfs_set(config):
    return [{'name': 'ADF%d' % i, 'func': None, 'arity': config['max_arity']} for i in range(config['num_adf'])]


def create_adfs_terminal_set(config):
    return [{'name': 'a%d' % i, 'value': 0, 'arity': 0} for i in range(config['max_arity'])]


def create_terminal_set(config):
    return [{'name': 'x%d' % _, 'value': 0, 'arity': 0} for _ in range(config['num_terminal'])]


def create_function_set():
    return FUNCTION_SET
