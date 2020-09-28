"""Make plot showing quantifications of performance of PAX on model Schlappa data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from manuscript_plots import set_plot_params

set_plot_params.init_paper_small()

import pax_simulation_pipeline
from manuscript_plots import schlappa_performance


def make_figure():
    log10_counts = schlappa_performance.LOG10_COUNTS_LIST
    data_list, num_counts = schlappa_performance._load_data(log10_counts)
    f, axs = plt.subplots(2, 1, sharex=True, figsize=(3.37, 3.37))
    _rmse_plot(axs[0], num_counts, data_list)
    _fwhm_plot(axs[1], num_counts, data_list)
    _format_figure(axs)


def _format_figure(axs):
    axs[0].set_ylabel("Norm. RMSE\n(a.u.)")
    axs[1].set_ylabel("FWHM of\nFirst Peak (meV)")
    axs[1].set_xlabel("Detected Electrons")
    plt.tight_layout()
    axs[0].text(
        0.9,
        0.8,
        "A",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=axs[0].transAxes,
    )
    axs[1].text(
        0.9,
        0.8,
        "B",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=axs[1].transAxes,
    )
    axs[1].axhline(83.25, linestyle="--", color="k")


def _rmse_plot(ax, num_electrons, data_list):
    norm_rmse_list = []
    for data in data_list:
        data = data["additional_deconvolutions"]
        mse_list = []
        for deconvolved in data:
            mse = mean_squared_error(
                deconvolved.deconvolved_y_, deconvolved.ground_truth_y
            )
            mse_list.append(mse)
        deconvolved_mse = np.mean(mse_list)
        rmse = np.sqrt(deconvolved_mse)
        norm_rmse = rmse / np.amax(data[0].ground_truth_y)
        norm_rmse_list.append(norm_rmse)
    ax.semilogx(num_electrons, norm_rmse_list, color="r", marker="o", markersize=4)


def _fwhm_plot(ax, num_electrons, data_list):
    fwhm_list = []
    for data in data_list:
        data = data["additional_deconvolutions"]
        current_fwhms = []
        for deconvolved in data:
            fwhm = _get_fwhm(deconvolved.deconvolved_x, deconvolved.deconvolved_y_)
            current_fwhms.append(fwhm)
        fwhm = np.mean(current_fwhms)
        fwhm_list.append(fwhm)
    ax.semilogx(
        num_electrons, 1e3 * np.array(fwhm_list), color="r", marker="o", markersize=4
    )


def _get_fwhm(deconvolved_x, deconvolved_y, center=0.0, width=1.0):
    """Return FWHM of loss peak at specified location
    """
    loss = deconvolved_x - 778
    loss_in_range = [(loss > (center - width / 2)) & (loss < (center + width / 2))]
    peak_location = loss[loss_in_range][np.argmax(deconvolved_y[loss_in_range])]
    peak_height = np.amax(deconvolved_y[loss_in_range])
    spec_below = deconvolved_y[loss < peak_location]
    loss_below = loss[loss < peak_location]
    above_peak = deconvolved_y[loss > peak_location]
    loss_above = loss[loss > peak_location]
    below_less_than_half = loss_below[spec_below < (peak_height / 2)]
    below_hwhm = peak_location - below_less_than_half[-1]
    above_less_than_half = loss_above[above_peak < (peak_height / 2)]
    above_hwhm = above_less_than_half[0] - peak_location
    fwhm = above_hwhm + below_hwhm
    return fwhm
