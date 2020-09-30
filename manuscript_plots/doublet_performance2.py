"""Make doublet performance figure with constant peak width
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from run_simulations import doublet2
from manuscript_plots import set_plot_params

set_plot_params.init_paper_small()

FIGURES_DIR = "figures"


def load_data():
    data = doublet2.load()
    return data


def doublet_performance2():
    data = doublet2.load()
    f = plt.figure(figsize=(3.37, 4.5))
    grid = plt.GridSpec(3, 2)
    ax_irf = f.add_subplot(grid[0, :])
    _irf_plot(ax_irf, data)
    ax_pax = f.add_subplot(grid[1:3, 0])
    ax_spectra = f.add_subplot(grid[1:3, 1], sharey=ax_pax)
    plt.setp(ax_spectra.get_yticklabels(), visible=False)
    _spectra_plot(ax_spectra, data)
    _pax_plot(ax_pax, data)
    _format_figure(f, ax_irf, ax_pax, ax_spectra, grid)
    file_name = f"{FIGURES_DIR}/pax_performance2.eps"
    plt.savefig(file_name, dpi=600)


def _irf_plot(ax, data_list):
    y = data_list[0]["deconvolved"][0].impulse_response_y
    y = y / np.amax(y)
    ax.plot(data_list[0]["deconvolved"][0].impulse_response_x, y, "k")


def _pax_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        deconvolved = data["deconvolved"][0]
        offset = ind * 1.0
        norm = 1.1 * np.amax(deconvolved.measured_y_)
        ax.plot(
            deconvolved.convolved_x,
            offset + deconvolved.reconstruction_y_ / norm,
            "r",
            label="Reconstruction",
        )
        ax.plot(
            deconvolved.convolved_x,
            offset + deconvolved.measured_y_ / norm,
            "k--",
            label="PAX",
        )


def _spectra_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        deconvolved = data["deconvolved"][0]
        energy_loss = -1 * (deconvolved.deconvolved_x - 778)
        offset = ind * 1.0
        norm = 1.1 * np.amax(deconvolved.ground_truth_y)
        ax.plot(
            energy_loss,
            offset + deconvolved.deconvolved_y_ / norm,
            "r",
            label="Deconvolved",
        )
        ax.plot(
            energy_loss,
            offset + deconvolved.ground_truth_y / norm,
            "k--",
            label="Ground Truth",
        )


def doublet_performance_old():
    data = load_data()
    _, axs = plt.subplots(3, 1, figsize=(3.37, 4), sharex=True)
    energy_loss = -1 * (data[0]["deconvolved"][0].deconvolved_x - 778)
    for i in range(len(data)):
        d_i = 2 - i
        for deconvolved in data[d_i]["deconvolved"]:
            axs[i].plot(
                energy_loss, deconvolved.deconvolved_y_ + 0.02, color="r", alpha=1
            )
        axs[i].plot(energy_loss, data[d_i]["ground_truth"]["y"] + 0.02, "--", color="k")
        for bootstrap in data[d_i]["bootstraps"]:
            axs[i].plot(energy_loss, bootstrap.deconvolved_y_, color="c", alpha=1)
        axs[i].plot(energy_loss, data[d_i]["ground_truth"]["y"], "--", color="k")
    format(axs)
    plt.axes((0.65, 0.75, 0.3, 0.2), facecolor="w")
    # plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.plot(
        data[0]["deconvolved"][0].impulse_response_x,
        data[0]["deconvolved"][0].impulse_response_y,
        color="k",
    )
    plt.gca().set_xlim((-0.1, 0.1))
    plt.gca().set_xlabel("Binding Energy\n(eV)")
    plt.gca().text(
        0.9,
        0.2,
        "D",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=plt.gca().transAxes,
    )
    plt.savefig("figures/doublet_performance2.eps", dpi=600)


def _format_figure(f, ax_irf, ax_pax, ax_spectra, grid):
    ax_irf.set_xlabel("Binding Energy (eV)")
    ax_irf.set_ylabel("Intensity (a.u.)")
    ax_irf.text(
        0.1,
        0.7,
        "A",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=ax_irf.transAxes,
    )
    ax_irf.text(
        0.97,
        0.65,
        "Model\nPhotoemission",
        fontsize=9,
        horizontalalignment="right",
        transform=ax_irf.transAxes,
    )
    ax_irf.set_xlim((-0.5, 0.5))
    legend_elements = [
        Line2D([0], [0], color="k", linestyle="--", label="PAX"),
        Line2D([0], [0], color="r", label="Reconstruction"),
    ]
    ax_pax.legend(handles=legend_elements, loc="upper left", frameon=False)
    ax_pax.set_ylabel("Intensity (a.u.)")
    ax_pax.set_xlabel("Kinetic Energy (eV)")
    ax_pax.text(
        0.9,
        0.76,
        "B",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=ax_pax.transAxes,
    )
    ax_pax.set_xlim((778.45, 778.65))
    ax_spectra.set_xlim((-0.05, 0.2))
    ax_spectra.invert_xaxis()
    ax_spectra.set_ylim((-0.2, 4.0))
    ax_spectra.set_xlabel("Energy Loss (eV)")
    legend_elements = [
        Line2D([0], [0], color="k", linestyle="--", label="Ground Truth"),
        Line2D([0], [0], color="r", label="Deconvolved"),
    ]
    ax_spectra.legend(handles=legend_elements, loc="upper left", frameon=False)
    ax_spectra.text(
        0.1,
        0.76,
        "C",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=ax_spectra.transAxes,
    )
    ax_irf.invert_xaxis()
    grid.tight_layout(f)
    texts = []
    # texts.append(ax_spectra.text(7.8, 2.4, '$N_e=10^'+str(int(spectra_log10_counts[2]))+'$', ha='center', transform=ax_spectra.transData))
    # texts.append(ax_spectra.text(7.8, 1.4, '$N_e=10^'+str(int(spectra_log10_counts[1]))+'$', ha='center', transform=ax_spectra.transData))
    # texts.append(ax_spectra.text(7.8, 0.4, '$N_e=10^'+str(int(spectra_log10_counts[0]))+'$', ha='center', transform=ax_spectra.transData))
    for text in texts:
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=5, foreground="white"),
                path_effects.Normal(),
            ]
        )


def format_old(axs):
    axs[0].set_xlim((-0.05, 0.12))
    for ax in axs:
        ax.invert_xaxis()
    axs[1].set_ylabel("Intensity (a.u.)")
    axs[2].set_xlabel("Energy Loss (eV)")
    plt.tight_layout()
    # Shrink current axis by 20%
    for ax in axs:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.55, box.height])
    # Put a legend to the right of the current axis
    custom_lines = [
        Line2D([0], [0], color="k", linestyle="--"),
        Line2D([0], [0], color="r"),
        Line2D([0], [0], color="c"),
        Line2D([0], [0], color="k"),
    ]
    axs[1].legend(
        custom_lines,
        [
            "Ground\nTruth",
            "Deconvolved\nSimulations",
            "Deconvolved\nBootstraps",
            "Model\nPhotoemission",
        ],
        loc="center left",
        bbox_to_anchor=(1.1, 0.05),
        frameon=True,
    )
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
    axs[2].text(
        0.9,
        0.8,
        "C",
        fontsize=10,
        weight="bold",
        horizontalalignment="center",
        transform=axs[2].transAxes,
    )
