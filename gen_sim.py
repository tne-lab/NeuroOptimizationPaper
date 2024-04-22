import math
import random

import numpy as np
import scipy
import sympy
from sympy import solve


def logit(x):
    return math.exp(x) / (1 + math.exp(x))


def normalize(x, x_range):
    return (x - x_range[0]) / (x_range[1] - x_range[0])


# Calculate sum of sinusoids for range of frequencies and amplitudes at given x value
def sos(freqs, amps, x):
    value = 0
    for i in range(len(amps[0])):
        value += amps[0][i] * math.cos(2 * math.pi * freqs[0][i] * x) + amps[1][i] * math.sin(2 * math.pi * freqs[1][i] * x)
    return value


# Combine sums of sinusoids for range of frequencies and amplitudes for multiple parameters at given x value
def combine_sos(sim, x, param_ranges):
    off_offset = 1.5
    off_scale = 1
    off_shift = 0.8
    value = 1
    for key in x:
        if x[key] < param_ranges[key][0] or x[key] > param_ranges[key][1]:
            return np.inf
    for param in sim['amps']:
        value *= sos(sim['freqs'][param], sim['amps'][param], normalize(x[param], param_ranges[param]))
    if any(x[key] == 0 for key in param_ranges):
        return 0
    else:
        if 'saf_a' in sim:
            if (x['amp'] - sim['saf_h']) ** 2 / sim['saf_a'] + sim['saf_k'] < x['pw']:
                return 0
            D = normalize(sim['D'](x['amp'], x['pw']), (0, math.sqrt(param_ranges['amp'][1] ** 2 + param_ranges['pw'][1] ** 2)))
            if D < 1 - off_shift:
                value /= (1 + (D ** off_offset / (1 - D - off_shift) ** off_offset) ** (-off_scale))
        for key in param_ranges:
            if normalize(x[key], param_ranges[key]) < 1 - off_shift:
                value /= (1 + (normalize(x[key], param_ranges[key]) ** off_offset / (1 - normalize(x[key], param_ranges[key]) - off_shift) ** off_offset) ** (-off_scale))
        return value


def gen_sim(effect, param_ranges, n_freq=10, max_freq=1, safety=False):
    sim = {'amps': {}, 'freqs': {}, 'param_ranges': param_ranges, 'max_freq': max_freq}
    # Create response surface for DDM parameters
    for param in param_ranges:
        sim['amps'][param] = ([random.random() - 0.5 for _ in range(n_freq)], [random.random() - 0.5 for _ in range(n_freq)])
        sim['freqs'][param] = ([random.random() * max_freq for _ in range(n_freq)], [random.random() * max_freq for _ in range(n_freq)])

    # Create safety region
    if safety:
        y = param_ranges['pw'][1]
        x = param_ranges['amp'][1]
        k = random.random() * y
        h = random.random() * (2000 - x) + x
        al = (x - h) ** 2 / (y - k)
        ah = h ** 2 / (y - k)
        a = random.random() * (ah - al) + al
        sim['saf_k'] = k
        sim['saf_a'] = a
        sim['saf_h'] = h
        xt = sympy.Symbol('xt')
        yt = sympy.Symbol('yt')
        xc = sympy.Symbol('x', real=True)
        xc = solve(sympy.diff((xc - xt) ** 2 + ((xc - sim['saf_h']) ** 2 / sim['saf_a'] + sim['saf_k'] - yt) ** 2, xc), xc)[0]
        sim['D'] = sympy.lambdify([xt, yt], sympy.sqrt((xc - xt) ** 2 + ((xc - sim['saf_h']) ** 2 / sim['saf_a'] + sim['saf_k'] - yt) ** 2))

    # Find global minimum of surface
    res_min = scipy.optimize.brute(
        lambda x: combine_sos(sim, {key: val for key, val in zip(param_ranges.keys(), x)}, param_ranges),
        ranges=tuple(param_ranges.values()), full_output=True, finish=lambda func, x0, args: scipy.optimize.fmin_l_bfgs_b(func, x0, approx_grad=True, bounds=list(param_ranges.values())))

    # Find global maximum of surface
    res_max = scipy.optimize.brute(
        lambda x: -combine_sos(sim, {key: val for key, val in zip(param_ranges.keys(), x)}, param_ranges),
        ranges=tuple(param_ranges.values()), full_output=True, finish=lambda func, x0, args: scipy.optimize.fmin_l_bfgs_b(func, x0, approx_grad=True, bounds=list(param_ranges.values())))

    # Sample DDM parameters from posterior
    sim['mean_effect'] = effect[0]
    sim['std_effect'] = effect[1]

    sim['scale'] = abs(res_max[1]) if abs(res_max[1]) > abs(res_min[1]) else res_min[1]

    if abs(res_max[1]) > abs(res_min[1]):
        sim['optimal_loc'] = res_max[0]
    else:
        sim['optimal_loc'] = res_min[0]

    return sim
