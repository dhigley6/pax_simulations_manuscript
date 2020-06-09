"""Make doublet performance figure with constant peak width
"""

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from run_simulations import doublet2
from manuscript_plots import set_plot_params
set_plot_params.init_paper_small()

def load_data():
    data = doublet2.load()
    return data

def doublet_performance2():
    data = load_data()
    _, axs = plt.subplots(3, 1, figsize=(3.37, 4), sharex=True)
    energy_loss = -1*(data[0]['deconvolved'][0].deconvolved_x-778)
    for i in range(len(data)):
        d_i = 2-i
        for deconvolved in data[d_i]['deconvolved']:
            axs[i].plot(energy_loss, deconvolved.deconvolved_y_+0.02, color='r', alpha=1)
        axs[i].plot(energy_loss, data[d_i]['ground_truth']['y']+0.02, '--', color='k')
        for bootstrap in data[d_i]['bootstraps']:
            axs[i].plot(energy_loss, bootstrap.deconvolved_y_, color='c', alpha=1)
        axs[i].plot(energy_loss, data[d_i]['ground_truth']['y'], '--', color='k')
    format(axs)
    plt.axes((0.65, 0.75, 0.3, 0.2), facecolor='w')
    #plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.plot(data[0]['deconvolved'][0].impulse_response_x, data[0]['deconvolved'][0].impulse_response_y, color='k')
    plt.gca().set_xlim((-0.1, 0.1))
    plt.gca().set_xlabel('Binding Energy\n(eV)')
    plt.gca().text(0.9, 0.2, 'D', fontsize=10, weight='bold', horizontalalignment='center',
       transform=plt.gca().transAxes)
    plt.savefig('figures/doublet_performance2.eps', dpi=600)

def format(axs):
    axs[0].set_xlim((-0.05, 0.12))
    for ax in axs:
        ax.invert_xaxis()
    axs[1].set_ylabel('Intensity (a.u.)')
    axs[2].set_xlabel('Energy Loss (eV)')
    plt.tight_layout()
    # Shrink current axis by 20%
    for ax in axs:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.55, box.height])
    # Put a legend to the right of the current axis
    custom_lines = [Line2D([0], [0], color='k', linestyle='--'),
                    Line2D([0], [0], color='r'),
                    Line2D([0], [0], color='c'),
                    Line2D([0], [0], color='k')]
    axs[1].legend(custom_lines, ['Ground\nTruth', 'Deconvolved\nSimulations', 'Deconvolved\nBootstraps', 'Model\nPhotoemission'],loc='center left', bbox_to_anchor=(1.1, 0.05),
                  frameon=True)
    axs[0].text(0.9, 0.8, 'A', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=axs[0].transAxes)
    axs[1].text(0.9, 0.8, 'B', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[1].transAxes)
    axs[2].text(0.9, 0.8, 'C', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[2].transAxes)