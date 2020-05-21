"""Run doublet simulations for manuscript
"""

import numpy as np 

import pax_simulation_pipeline
from manuscript_plots import doublet_performance

PARAMETERS = {
    'energy_loss': np.arange(-0.2, 0.4, 0.002),
    'iterations': 1E5,
    'simulations': 1000,
    'cv_fold': 3,
    'regularization_strengths': np.logspace(-4, -2, 10)
}
TOTAL_SEPARATION_LIST = doublet_performance.TOTAL_SEPARATION_LIST
TOTAL_LOG10_NUM_ELECTRONS_LIST = doublet_performance.TOTAL_LOG10_NUM_ELECTRONS_LIST

def run_fermi():
    for separation in TOTAL_SEPARATION_LIST:
        for log10_num_electrons in TOTAL_LOG10_NUM_ELECTRONS_LIST:
            _ = pax_simulation_pipeline.run(log10_num_electrons, 
                                            rixs=['doublet', separation], 
                                            photoemission='fermi', **PARAMETERS)
        print('Completed separation '+str(separation))