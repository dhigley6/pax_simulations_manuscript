"""Implement analysis procedure proposed in Dakovski et al. (2017)

A bit hacked together since I'm only planning on using this once.
"""

from operator import mul
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.optimize import least_squares
from scipy.signal import convolve

from manuscript_plots.lcls import pax_lcls2016
from pax_deconvolve.deconvolution import deconvolvers

from manuscript_plots import set_plot_params

set_plot_params.init_paper_small()

PHOTON_ENERGY_OFFSET = 804.23-23.8    # (eV) (determined empirically)
KE_OFFSET = 12.66    # (eV) (determine empirically)


#GAUSS_CENTERS = [0, 0.2, 0.4, 0.6, 0.8, 1, 15]
GAUSS_CENTERS = [0, 4, 7, 8, 9, 10, 11, 12]

def fit_curve(x, y):

    def test_func(center):
        gauss = np.exp(-(x-center)**2)
        difference = gauss-y
        return np.sqrt(np.mean(difference**2))

    
    result = least_squares(test_func, [0])
    return result

def get_reconstruction(deconvolved_y, impulse_response_y):
    reconstruction = convolve(
        deconvolved_y,
        impulse_response_y,
        mode='valid',
    )
    return reconstruction

def test_run():
    x = np.linspace(-10, 10, 100)
    y = np.exp(-(x-0.5)**2)
    result = fit_curve(x, y)
    plt.figure()
    plt.plot(x, y)
    func_estimate = np.exp(-(x-result.x[0])**2)
    plt.plot(x, func_estimate)

def test_single_gauss():
    lcls_data = pax_lcls2016.get_lcls_specs()
    impulse_response = lcls_data['psf']
    measured_pax = lcls_data['spectra'][0]

    def rmse_from_params(params): 
        return rmse_single_gauss(params, measured_pax['x'], impulse_response, measured_pax)

    params_guess = [703, 1, 0.01]
    result = least_squares(rmse_from_params, params_guess)
    params = result.x
    print(params)
    deconvolved_y = single_gauss(params, measured_pax['x'])
    reconstructed_y = get_reconstruction(deconvolved_y, impulse_response['y'])
    f, axs = plt.subplots(1, 2)
    axs[0].plot(measured_pax['x'], deconvolved_y)
    axs[1].plot(measured_pax['x'], measured_pax['y'])
    axs[1].plot(measured_pax['x'], reconstructed_y)

def test_all_lcls():
    results = []
    for number in range(9):
        result = test_multiple_gauss(number)
        results.append(result)
    f, axs = plt.subplots(1, 2, figsize=(3.37, 4.5))
    for ind, deconvolved in enumerate(results):
        norm = np.amax(deconvolved['measured_pax']['y'])
        axs[0].plot(deconvolved['measured_pax']['x']-KE_OFFSET, deconvolved['measured_pax']['y']/norm-1.1*ind, 'k--', label='PAX')
        axs[0].plot(deconvolved['reconstructed']['x']-KE_OFFSET, deconvolved['reconstructed']['y']/norm-1.1*ind, 'r', label='Reconstruction')
    plt.figure()
    for ind, deconvolved in enumerate(results):
        norm = np.amax(deconvolved['deconvolved']['y'])
        axs[1].plot(deconvolved['deconvolved']['x'], deconvolved['deconvolved']['y']/norm-1.1*ind, 'r', label='deconvolved')
    plt.setp(axs[1].get_yticklabels(), visible=False)
    axs[0].set_xlabel('Kinetic Energy (eV)')
    axs[0].set_ylabel('Intensity (a. u.)')
    axs[1].set_xlabel('Energy Loss (eV)')
    textprops = {
        'fontsize': 10,
        'weight': 'bold',
        'horizontalalignment': 'center',
    }
    axs[0].text(0.1, 0.85, 'A', transform=axs[0].transAxes, **textprops)
    axs[1].text(0.9, 0.85, 'B', transform=axs[1].transAxes, **textprops)
    #axs[0].set_xlim((678, 707))
    axs[0].set_ylim((-9, 2.5))
    axs[1].set_xlim((-5, 15))
    axs[1].set_ylim((-9, 2.5))
    axs[1].invert_xaxis()
    legend_elements = [
        Line2D([0], [0], color='k', linestyle='--', label='PAX'),
        Line2D([0], [0], color='r', label='Reconstruction')
    ]
    axs[0].legend(handles=legend_elements, loc='upper left', frameon=False)
    legend_elements = [
        Line2D([0], [0], color="r", label="Deconvolved"),
    ]
    axs[1].legend(handles=legend_elements, loc="upper left", frameon=False)
    plt.gcf().tight_layout()

def test_multiple_gauss(number=0):
    lcls_data = pax_lcls2016.get_lcls_specs()
    # pop off duplicate spectrum
    lcls_data['spectra'].pop(7)
    lcls_data['incident_photon_energy'].pop(7)
    incident_photon_energy = lcls_data['incident_photon_energy'][number]
    impulse_response = lcls_data['psf']
    measured_pax = lcls_data['spectra'][number]
    deconvolved_x = deconvolvers._get_deconvolved_x(measured_pax['x'], impulse_response['x'])
    energy_loss = get_energy_loss(deconvolved_x, incident_photon_energy)

    def resid_from_params(params):
        return resid_multiple_gauss(params, energy_loss, impulse_response, measured_pax)

    params_guess = [0.1, 0.01]*len(GAUSS_CENTERS)
    result = least_squares(resid_from_params, params_guess, method='lm')
    params = result.x
    print(params)
    deconvolved_y = multiple_gauss(params, energy_loss)
    reconstructed_y = get_reconstruction(deconvolved_y, impulse_response['y'])
    f, axs = plt.subplots(1, 2)
    axs[0].plot(energy_loss, deconvolved_y)
    axs[1].plot(measured_pax['x'], measured_pax['y'])
    axs[1].plot(measured_pax['x'], reconstructed_y)
    deconvolved_result = {
        'x': energy_loss,
        'y': deconvolved_y
    }
    reconstructed_result = {
        'x': measured_pax['x'],
        'y': reconstructed_y
    }
    to_return = {
        'deconvolved': deconvolved_result,
        'measured_pax': measured_pax,
        'reconstructed': reconstructed_result
    }
    return to_return

def multiple_gauss(params, x):
    y = np.zeros_like(x)
    params_reshaped = np.reshape(params, (-1, 2))
    for ind, row in enumerate(params_reshaped):
        full_params = np.hstack([GAUSS_CENTERS[ind], row])
        single_y = single_gauss(full_params, x)
        y = y + single_y
    return y

def resid_multiple_gauss(params, x, impulse_response, measured_pax):
    y = multiple_gauss(params, x)
    reconstructed_pax_y = get_reconstruction(y, impulse_response['y'])
    resid = reconstructed_pax_y-measured_pax['y']
    return resid

def single_gauss(params, x):
    center = params[0]
    width = params[1]
    amplitude = params[2]
    y = amplitude*np.exp(-(x-center)**2/(width**2))
    return y

def rmse_single_gauss(params, x, impulse_response, measured_pax):
    gauss_y = single_gauss(params, x)
    reconstructed_pax_y = get_reconstruction(gauss_y, impulse_response['y'])
    rmse = np.sqrt(np.mean((reconstructed_pax_y-measured_pax['y'])**2))
    plt.figure()
    plt.plot(reconstructed_pax_y)
    plt.plot(measured_pax['y'])
    return rmse

def get_energy_loss(deconvolved_x, incident_photon_energy):
    energy_loss = -1*(deconvolved_x-incident_photon_energy+PHOTON_ENERGY_OFFSET)
    return energy_loss