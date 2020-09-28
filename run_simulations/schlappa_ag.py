"""Running PAX simulations with Schlappa model RIXS and Ag converter
"""

from manuscript_plots import schlappa_performance
import pax_simulation_pipeline

SCHLAPPA_PARAMETERS = schlappa_performance.SCHLAPPA_PARAMETERS


def run_simulations():
    """Run analysis for making figures
    (used by schlappa_performance.py and schlappa_performance_quant.py)
    """
    print("Running Schlappa RIXS, Ag converter simulations")
    for log10_num_electrons in schlappa_performance.LOG10_COUNTS_LIST:
        _ = pax_simulation_pipeline.run(
            log10_num_electrons,
            rixs="schlappa",
            photoemission="ag",
            num_additional=25,
            **SCHLAPPA_PARAMETERS
        )
        print("Completed " + str(log10_num_electrons))
