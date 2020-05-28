"""Make plot showing simulated PAX performance on model Schlappa RIXS
with Ag 3d converter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error
import pickle

import pax_simulation_pipeline
from manuscript_plots import set_plot_params
set_plot_params.init_paper_small()

FIGURES_DIR = 'figures'
# List of base 10 logarithm of detected electrons to simulate:
LOG10_COUNTS_LIST = [7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5]

# Below are the parameters to run the simulations with
SCHLAPPA_PARAMETERS = {
    'energy_loss': np.arange(-8, 10, 0.01),
    'iterations': int(1E5),
    'simulations': 1000,
    'cv_fold': 3,
    'regularizer_widths': np.logspace(-3, -1, 10)
}

def run_schlappa_ag_with_extra_analysis():
    """Run analysis for making figure
    """
    for log10_num_electrons in [6.5]:
        _ = pax_simulation_pipeline.run_with_extra(log10_num_electrons, rixs='schlappa', photoemission='ag', num_analyses=25, **SCHLAPPA_PARAMETERS)
        print('Completed '+str(log10_num_electrons))

def make_figure():
    log10_counts = LOG10_COUNTS_LIST
    data_list = []
    num_counts = []
    for i in log10_counts:
        data_list.append(_load_old(i, rixs='schlappa', photoemission='ag'))
        num_counts.append(10**i)
    spectra_log10_counts = [7.0, 5.0, 3.0]
    spectra_data_list = []
    spectra_num_counts = []
    for i in spectra_log10_counts:
        spectra_data_list.append(_load_old(i, rixs='schlappa', photoemission='ag'))
        spectra_num_counts.append(i)
    f = plt.figure(figsize=(3.37, 5.5))
    grid = plt.GridSpec(4, 2)
    ax_spectra = f.add_subplot(grid[:2, :])
    ax_deconvolved_mse = f.add_subplot(grid[2, :])
    ax_fwhm = f.add_subplot(grid[3, :], sharex=ax_deconvolved_mse)
    _spectra_plot(ax_spectra, spectra_data_list)
    _rmse_plot(ax_deconvolved_mse, num_counts, data_list)
    _fwhm_plot(ax_fwhm, num_counts, data_list)
    axs = [ax_spectra, ax_deconvolved_mse, ax_fwhm]
    _format_figure(axs, spectra_num_counts)
    file_name = f'{FIGURES_DIR}/pax_performance1.eps'
    plt.savefig(file_name, dpi=600)
    
def _spectra_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        deconvolved = data['cv_deconvolver']
        energy_loss = -1*(deconvolved.deconvolved_x-778)
        offset = ind*1.0
        norm = 1.1*np.amax(deconvolved.ground_truth_y)
        ax.plot(energy_loss, offset+deconvolved.deconvolved_y_/norm, 'r', label='Deconvolved')
        ax.plot(energy_loss, offset+deconvolved.ground_truth_y/norm, 'k--', label='Ground Truth')
        
def _rmse_plot(ax, num_electrons, data_list):
    norm_rmse_list = []
    for data in data_list:
        data = data['analyses']
        mse_list = []
        for deconvolved in data:
            mse = mean_squared_error(deconvolved.deconvolved_y_, deconvolved.ground_truth_y)
            mse_list.append(mse)
        deconvolved_mse = np.mean(mse_list)
        rmse = np.sqrt(deconvolved_mse)
        norm_rmse = rmse/np.amax(data[0].ground_truth_y)
        norm_rmse_list.append(norm_rmse)
    ax.semilogx(num_electrons, norm_rmse_list, color='r', marker='o', markersize=4)
    
def _fwhm_plot(ax, num_electrons, data_list):
    fwhm_list = []
    for data in data_list:
        data = data['analyses']
        current_fwhms = []
        for deconvolved in data:
            fwhm = _get_fwhm(deconvolved.deconvolved_x, deconvolved.deconvolved_y_)
            current_fwhms.append(fwhm)
        fwhm = np.mean(current_fwhms)
        fwhm_list.append(fwhm)
    ax.semilogx(num_electrons, 1E3*np.array(fwhm_list), color='r', marker='o', markersize=4)
    ax.axhline(83.25, linestyle='--', color='k')

def _get_fwhm(deconvolved_x, deconvolved_y, center=0.0, width=1.0):
    """Return FWHM of loss peak at specified location
    """
    loss = deconvolved_x-778
    loss_in_range = [(loss > (center-width/2)) & (loss < (center+width/2))]
    peak_location = loss[loss_in_range][np.argmax(deconvolved_y[loss_in_range])]
    peak_height = np.amax(deconvolved_y[loss_in_range])
    spec_below = deconvolved_y[loss < peak_location]
    loss_below = loss[loss < peak_location]
    above_peak = deconvolved_y[loss > peak_location]
    loss_above = loss[loss > peak_location]
    below_less_than_half = loss_below[spec_below < (peak_height/2)]
    below_hwhm = peak_location-below_less_than_half[-1]
    above_less_than_half = loss_above[above_peak < (peak_height/2)]
    above_hwhm = above_less_than_half[0]-peak_location
    fwhm = above_hwhm+below_hwhm
    return fwhm

def _format_figure(axs, spectra_counts):
    axs[0].set_xlim((-1, 7))
    axs[0].invert_xaxis()
    axs[0].set_ylim((-0.2, 3.5))
    axs[1].set_ylim((0, 0.06))
    axs[2].set_ylim((0, 450))
    axs[0].set_xlabel('Energy Loss (eV)')
    axs[0].set_ylabel('Intensity (a.u.)')
    axs[1].set_ylabel('Norm. RMS\nError (a.u.)')
    axs[2].set_ylabel('FWHM of\nFirst Peak (meV)')
    axs[2].set_xlabel('Detected Electrons')
    plt.setp(axs[1].get_xticklabels(), visible=False)
    legend_elements = [Line2D([0], [0], color='k', linestyle='--', label='Ground Truth'),
                       Line2D([0], [0], color='r', label='Deconvolved')]
    axs[0].legend(handles=legend_elements, loc='upper left', frameon=False)
    axs[0].text(-0, 2.4, '$N_e=10^'+str(int(spectra_counts[2]))+'$', ha='center', transform=axs[0].transData)
    axs[0].text(-0, 1.4, '$N_e=10^'+str(int(spectra_counts[1]))+'$', ha='center', transform=axs[0].transData)
    axs[0].text(-0, 0.4, '$N_e=10^'+str(int(spectra_counts[0]))+'$', ha='center', transform=axs[0].transData)
    plt.tight_layout()
    axs[0].text(0.9, 0.9, 'A', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=axs[0].transAxes)
    axs[1].text(0.9, 0.8, 'B', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[1].transAxes)
    axs[2].text(0.9, 0.8, 'C', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[2].transAxes)

def _load_old(log10_num_electrons, rixs='schlappa', photoemission='ag'):
    """Load old data set
    """
    file_name = _get_old_filename(log10_num_electrons, rixs, photoemission)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def _get_old_filename(log10_num_electrons, rixs, photoemission):
    file_name = '{}/{}_{}_rixs_1E{}_with_extra.pickle'.format(
        'old_simulated_results',
        photoemission,
        rixs,
        log10_num_electrons)
    return file_name