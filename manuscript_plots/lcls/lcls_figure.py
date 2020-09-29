"""Make plot showing analysis of 2016 LCLS data with algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

from manuscript_plots.lcls import pax_lcls2016
from manuscript_plots import set_plot_params

from pax_deconvolve.deconvolution import deconvolvers
from pax_deconvolve import visualize

set_plot_params.init_paper_small()

PHOTON_ENERGY_OFFSET = 804.23    # (eV) (determined empirically)

def lcls_figure():
    specs = pax_lcls2016.get_lcls_specs()
    regularization_strength = _estimate_best_regularization_strength(specs)
    print(f'Using regularization strength of {regularization_strength} eV')
    # take out extra 780 eV spectrum:
    specs['spectra'].pop(7)
    specs['incident_photon_energy'].pop(7)
    deconvolver_list = _deconvolve_spectra(specs, regularization_strength)
    f = plt.figure(figsize=(3.37, 5))
    grid = plt.GridSpec(4, 2)
    ax_irf = f.add_subplot(grid[0, :])
    ax_pax = f.add_subplot(grid[1:4, 0])
    ax_spectra = f.add_subplot(grid[1:4, 1])
    plt.setp(ax_spectra.get_yticklabels(), visible=False)
    _irf_plot(ax_irf, specs)
    _pax_plot(ax_pax, specs, deconvolver_list)
    _spectra_plot(ax_spectra, specs, deconvolver_list)
    _format_figure(ax_irf, ax_pax, ax_spectra, specs)

def _irf_plot(ax_irf, specs):
    norm = np.amax(specs['psf']['y'])
    ax_irf.plot(specs['psf']['binding_energy'], specs['psf']['y']/norm, 'k')

def _pax_plot(ax_pax, specs, deconvolver_list):
    for ind, deconvolved in enumerate(deconvolver_list):
        norm = np.amax(deconvolved.measured_y_)
        ax_pax.plot(deconvolved.convolved_x, -1.1*ind+deconvolved.measured_y_/norm, 'k--', label='PAX')
        ax_pax.plot(deconvolved.convolved_x, -1.1*ind+deconvolved.reconstruction_y_/norm, 'r', label='Reconstruction')


def _spectra_plot(ax_spectra, specs, deconvolver_list):
    for ind, deconvolved in enumerate(deconvolver_list):
        incident_photon_energy = specs['incident_photon_energy'][ind]
        label = ''.join([str(incident_photon_energy), ' eV'])
        norm = np.amax(deconvolved.deconvolved_y_)
        energy_loss = -1*(deconvolved.deconvolved_x-incident_photon_energy+PHOTON_ENERGY_OFFSET)
        ax_spectra.plot(energy_loss, -1.1*ind+deconvolved.deconvolved_y_/norm, 'r', label='Deconvolved')

def _format_figure(ax_irf, ax_pax, ax_spectra, specs):
    ax_irf.set_xlabel('Binding Energy (eV)')
    ax_irf.set_ylabel('Intensity (a.u.)')
    ax_pax.set_xlabel('Kinetic Energy (eV)')
    ax_pax.set_ylabel('Intensity (a.u.)')
    ax_spectra.set_xlabel('Energy Loss (eV)')
    ax_irf.text(
        0.03,
        0.6,
        "Measured\nPhotoemission",
        fontsize=9,
        horizontalalignment="left",
        transform=ax_irf.transAxes,
    )
    textprops = {
        'fontsize': 10,
        'weight': 'bold',
        'horizontalalignment': 'center',
    }
    ax_irf.text(0.9, 0.8, 'A', transform=ax_irf.transAxes, **textprops)
    ax_pax.text(0.1, 0.85, 'B', transform=ax_pax.transAxes, **textprops)
    ax_spectra.text(0.9, 0.85, 'C', transform=ax_spectra.transAxes, **textprops)
    ax_irf.set_xlim((65, 88))
    ax_pax.set_xlim((690, 720))
    ax_pax.set_ylim((-9, 2.5))
    ax_spectra.set_xlim((-5, 15))
    ax_spectra.set_ylim((-9, 2.5))
    ax_irf.invert_xaxis()
    ax_spectra.invert_xaxis()
    legend_elements = [
        Line2D([0], [0], color='k', linestyle='--', label='PAX'),
        Line2D([0], [0], color='r', label='Reconstruction')
    ]
    ax_pax.legend(handles=legend_elements, loc='upper left', frameon=False)
    legend_elements = [
        Line2D([0], [0], color="r", label="Deconvolved"),
    ]
    ax_spectra.legend(handles=legend_elements, loc="upper left", frameon=False)
    plt.gcf().tight_layout()
    for ind, spec in enumerate(specs['spectra']):
        incident_photon_energy = specs['incident_photon_energy'][ind]
        offset = -1.1*ind
        label = ''.join([str(incident_photon_energy), ' eV'])
        text = ax_spectra.text(17.5, offset+0.3, label, ha='center', transform=ax_spectra.transData)
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=5, foreground='white'),
                path_effects.Normal(),
            ]
        )
    plt.savefig('figures/lcls_result.eps', dpi=600)

def _deconvolve_spectra(specs, regularization_strength):
    deconvolver_list = []
    for ind, spec in enumerate(specs['spectra']):
        deconvolver = deconvolvers.LRFisterDeconvolve(
            specs['psf']['x'],
            specs['psf']['y']/np.sum(specs['psf']['y']),
            specs['spectra'][ind]['x'],
            regularization_strength=regularization_strength
        )
        measured_y = np.array([spec['y']])
        _ = deconvolver.fit(measured_y)
        deconvolver_list.append(deconvolver)
    return deconvolver_list

def _estimate_best_regularization_strength(specs):
    deconvolver = deconvolvers.LRFisterGrid(
        specs['psf']['x'],
        specs['psf']['y']/np.sum(specs['psf']['y']),
        specs['spectra'][0]['x'],
        cv_folds=2
    )
    to_fit = np.array([specs['spectra'][6]['y'], specs['spectra'][7]['y']])
    _ = deconvolver.fit(to_fit)
    visualize.plot_result(deconvolver)
    return deconvolver.best_regularization_strength_