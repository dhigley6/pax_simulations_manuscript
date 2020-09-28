"""Make plot showing simulated PAX performance on model Schlappa RIXS
with Ag 3d converter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from sklearn.metrics import mean_squared_error
import pickle

import pax_simulation_pipeline
from manuscript_plots import set_plot_params
set_plot_params.init_paper_small()

FIGURES_DIR = 'figures'
# List of base 10 logarithm of detected electrons to simulate:
LOG10_COUNTS_LIST = [7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5]
#LOG10_COUNTS_LIST = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5]

# Below are the parameters to run the simulations with
SCHLAPPA_PARAMETERS = {
    'energy_loss': np.arange(-8, 10, 0.01),
    'iterations': int(1E4),
    'simulations': 1000,
    'cv_fold': 3,
    'regularizer_widths': np.logspace(-3, -1, 10)
}

def _load_data(log10_counts_to_load):
    data_list = []
    num_counts = []
    for i in log10_counts_to_load:
        data_list.append(pax_simulation_pipeline.load(i, rixs='schlappa', photoemission='ag'))
        num_counts.append(10**i)
    return data_list, num_counts

def make_figure():
    data_list, num_counts = _load_data(LOG10_COUNTS_LIST)
    spectra_log10_counts = [7.0, 5.0, 3.0]
    spectra_data_list, spectra_num_counts = _load_data(spectra_log10_counts)
    a = 1/0
    f = plt.figure(figsize=(3.37, 4.5))
    grid = plt.GridSpec(3, 2)
    ax_irf = f.add_subplot(grid[0, :])
    _irf_plot(ax_irf, data_list)
    ax_pax = f.add_subplot(grid[1:3, 0])
    ax_spectra = f.add_subplot(grid[1:3, 1], sharey=ax_pax)
    plt.setp(ax_spectra.get_yticklabels(), visible=False)
    _spectra_plot(ax_spectra, spectra_data_list, spectra_num_counts)
    _pax_plot(ax_pax, spectra_data_list)
    _format_figure(f, ax_irf, ax_pax, ax_spectra, grid, spectra_log10_counts)
    file_name = f'{FIGURES_DIR}/pax_performance1.eps'
    plt.savefig(file_name, dpi=600)

def _format_figure(f, ax_irf, ax_pax, ax_spectra, grid, spectra_log10_counts):
    ax_irf.set_xlabel('Binding Energy (eV)')
    ax_irf.set_ylabel('Intensity (a.u.)')
    ax_irf.text(0.9, 0.8, 'A', fontsize=10, weight='bold', horizontalalignment='center',
       transform=ax_irf.transAxes)
    ax_irf.text(0.03, 0.8, 'Model Photoemission', fontsize=9, horizontalalignment='left',
       transform=ax_irf.transAxes)
    ax_irf.set_xlim((365, 380))
    legend_elements = [Line2D([0], [0], color='k', linestyle='--', label='PAX'),
                       Line2D([0], [0], color='r', label='Reconstruction')]
    ax_pax.legend(handles=legend_elements, loc='upper left', frameon=False)
    ax_pax.set_ylabel('Intensity (a.u.)')
    ax_pax.set_xlabel('Kinetic Energy (eV)')
    ax_pax.text(0.1, 0.76, 'B', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=ax_pax.transAxes)
    ax_pax.set_xlim((415, 430))
    ax_spectra.invert_xaxis()
    ax_spectra.set_ylim((-0.2, 3.5))
    ax_spectra.set_xlabel('Energy Loss (eV)')
    legend_elements = [Line2D([0], [0], color='k', linestyle='--', label='Ground Truth'),
                       Line2D([0], [0], color='r', label='Deconvolved')]
    ax_spectra.legend(handles=legend_elements, loc='upper left', frameon=False)
    ax_spectra.text(0.9, 0.76, 'C', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=ax_spectra.transAxes)
    grid.tight_layout(f)
    texts = []
    texts.append(ax_spectra.text(7.8, 2.4, '$N_e=10^'+str(int(spectra_log10_counts[2]))+'$', ha='center', transform=ax_spectra.transData))
    texts.append(ax_spectra.text(7.8, 1.4, '$N_e=10^'+str(int(spectra_log10_counts[1]))+'$', ha='center', transform=ax_spectra.transData))
    texts.append(ax_spectra.text(7.8, 0.4, '$N_e=10^'+str(int(spectra_log10_counts[0]))+'$', ha='center', transform=ax_spectra.transData))
    for text in texts:
        text.set_path_effects([path_effects.Stroke(linewidth=5, foreground='white'),
                               path_effects.Normal()])


def _irf_plot(ax, data_list):
    y = data_list[0]['cv_deconvolver'].impulse_response_y
    y = y/np.amax(y)
    ax.plot(-1*data_list[0]['cv_deconvolver'].impulse_response_x, y, 'k')

def _pax_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        deconvolved = data['cv_deconvolver']
        offset = ind*1.0
        norm = 1.1*np.amax(deconvolved.measured_y_)
        if ind in [0]:
            # plot PAX second
            ax.plot(deconvolved.convolved_x, offset+deconvolved.reconstruction_y_/norm, 'r', label='Reconstruction')
            ax.plot(deconvolved.convolved_x, offset+deconvolved.measured_y_/norm, 'k--', label='PAX') 
        else:
            # plot PAX first
            ax.plot(deconvolved.convolved_x, offset+deconvolved.measured_y_/norm, 'k--', label='PAX') 
            ax.plot(deconvolved.convolved_x, offset+deconvolved.reconstruction_y_/norm, 'r', label='Reconstruction')
    

def _spectra_plot(ax, data_list, spectra_counts):
    for ind, data in enumerate(data_list):
        deconvolved = data['cv_deconvolver']
        energy_loss = -1*(deconvolved.deconvolved_x-778)
        offset = ind*1.0
        norm = 1.1*np.amax(deconvolved.ground_truth_y)
        ax.plot(energy_loss, offset+deconvolved.deconvolved_y_/norm, 'r', label='Deconvolved')
        ax.plot(energy_loss, offset+deconvolved.ground_truth_y/norm, 'k--', label='Ground Truth')
    ax.set_xlim((-1, 7))