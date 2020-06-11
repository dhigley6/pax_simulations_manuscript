"""Make plot showing how we approximate that the desired level of convergence has been reached
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pickle

from manuscript_plots import set_plot_params
set_plot_params.init_paper_small()
from pax_deconvolve.pax_simulations import simulate_pax
from pax_deconvolve.deconvolution import assess_convergence

FIGURES_DIR = 'figures'
REGULARIZATION_PARAMETERS = [0.0028, 0.0129, 0.1] # regularization parameters to plot

def run_ana():
    """Run analysis to create data used in this plot
    """
    log10_num_electrons = 4.0
    rixs_model = 'schlappa'
    photoemission_model = 'ag'
    num_simulations = 100
    energy_loss = np.arange(-8, 10, 0.01)
    regularization_strengths = np.logspace(-3, -1, 10)
    iterations = 1000
    impulse_response, pax_spectra, xray_xy = simulate_pax.simulate_from_presets(
        log10_num_electrons,
        rixs_model,
        photoemission_model,
        num_simulations,
        energy_loss
    )
    assess_convergence.run(
        impulse_response,
        pax_spectra,
        xray_xy,
        regularization_strengths,
        iterations
    )

def make_figure():
    _, axs = plt.subplots(2, 1, sharex=True, figsize=(3.37, 3.5))
    deconvolved_list = _load_deconvolved_mse()
    _make_deconvolved_mse_plot(axs[0], deconvolved_list)
    val_list = _load_val_mse()
    _make_val_mse_plot(axs[1], val_list)
    _format_figure(axs)
    file_name = f'{FIGURES_DIR}/convergence.eps'
    plt.savefig(file_name, dpi=600)

def _load_deconvolved_mse():
    data_list = []
    for par in REGULARIZATION_PARAMETERS:
        file_name = f'data/convergence_deconvolved/run-{str(par)}.csv'
        data = np.genfromtxt(file_name, skip_header=1, delimiter=',')
        data_list.append(data)
    return data_list

def _load_val_mse():
    data_list = []
    for par in REGULARIZATION_PARAMETERS:
        file_name = f'data/convergence_validation_reconstruction/run-{str(par)}.csv'
        data = np.genfromtxt(file_name, skip_header=1, delimiter=',')
        data_list.append(data)
    return data_list

def _make_deconvolved_mse_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        ax.loglog(data[:, 1], data[:, 2], label=str(1000*REGULARIZATION_PARAMETERS[ind]))

def _make_val_mse_plot(ax, data_list):
    min_list = []
    for data in data_list:
        minimum = np.amin(data[:, 2])
        min_list.append(minimum)
    minimum = np.amin(min_list)
    for ind, data in enumerate(data_list):
        ax.loglog(data[:, 1], data[:, 2]-minimum+10**(-2), label=str(REGULARIZATION_PARAMETERS[ind]))
    ax.axvline(18*4, color='k', linestyle='--')

def _format_figure(axs):
    axs[1].annotate('stopping\ncriterion met', (18*4, 0.5), xytext=(5, 0.5), arrowprops=dict(arrowstyle="->"), verticalalignment='center')
    axs[0].set_ylim((0.35, 25))
    axs[1].set_xlim((1, 1E3))
    axs[0].legend(loc='upper center', ncol=2, title='Regularization\nStrength (meV)')
    axs[0].set_ylabel('Deconvolved\nMSE (a.u.)')
    axs[1].set_ylabel('Val. Reconstruction\nMSE-minimum+10$^{-2}$ (a.u.)')
    axs[1].set_xlabel('Iterations')
    axs[0].text(0.1, 0.05, 'A', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=axs[0].transAxes)
    axs[1].text(0.1, 0.05, 'B', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[1].transAxes)
    plt.tight_layout()