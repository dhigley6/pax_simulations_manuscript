"""Run doublet simulations with independently set peak width and peak separation
"""

import numpy as np 
import random
import pickle

from pax_deconvolve.deconvolution import deconvolvers
from pax_deconvolve.pax_simulations import simulate_pax

LOG10_COUNTS_LIST = [7.0]
SEPARATIONS = [0.025, 0.045, 0.07]
NUM_SIMULATIONS = 3
NUM_BOOTSTRAPS = 3
ITERATIONS = 1E5

def load():
    data_list = []
    for separation in SEPARATIONS:
        data = load_set(separation, 7.0)
        data_list.append(data)
    return data_list


def load_set(separation, log10_counts):
    file_name = 'simulated_results/doublet2_'+str(separation)+'_'+str(log10_counts)+'.pickle'
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def run():
    for separation in SEPARATIONS:
        for log10_counts in LOG10_COUNTS_LIST:
            run_set(separation, log10_counts)

def run_set(separation, log10_counts):
    deconvolved_list = []
    for i in range(NUM_SIMULATIONS):
        impulse_response, pax_spectra, xray_xy = simulate_pax.simulate_from_presets(
            log10_counts,
            ['i_doublet', separation],
            'fermi',
            1000,
            np.arange(-0.2, 0.4, 0.002)
        )
        deconvolver = deconvolvers.LRFisterGrid(
            impulse_response['x'],
            impulse_response['y'],
            pax_spectra['x'],
            np.logspace(-4, -2, 10),
            ITERATIONS,
            xray_xy['y']
        )
        _ = deconvolver.fit(np.array(pax_spectra['y']))
        deconvolved_list.append(deconvolver)
        print(f'Completed {str(log10_counts)} counts, {str(separation)} separation, {str(i)} iteration')
    bootstrap_results = _run_bootstraps(impulse_response, pax_spectra, xray_xy, deconvolver.best_regularization_strength_)
    to_save = {
        'deconvolved': deconvolved_list,
        'bootstraps': bootstrap_results,
        'ground_truth': xray_xy
    }
    file_name = 'simulated_results/doublet2_'+str(separation)+'_'+str(log10_counts)+'.pickle'
    with open(file_name, 'wb') as f:
        pickle.dump(to_save, f)
    

def _run_bootstraps(impulse_response, pax_spectra, xray_xy, regularization_strength):
    bootstrap_deconvolved_list = []
    for _ in range(NUM_BOOTSTRAPS):
        bootstrapped_pax = {
            'x': pax_spectra['x'],
            'y': bootstrap_pax_set(pax_spectra['y'])
        }
        deconvolver = deconvolvers.LRFisterDeconvolve(
            impulse_response['x'],
            impulse_response['y'],
            bootstrapped_pax['x'],
            regularization_strength,
            ITERATIONS,
            xray_xy['y']
        )
        _ = deconvolver.fit(np.array(bootstrapped_pax['y']))
        bootstrap_deconvolved_list.append(deconvolver)
        print('Completed bootstrap '+str(_))
    return bootstrap_deconvolved_list
    

def bootstrap_pax_set(pax_spectra_y):
    bootstrapped_y = random.choices(pax_spectra_y, k=len(pax_spectra_y))
    return bootstrapped_y