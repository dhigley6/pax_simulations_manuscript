#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:55:10 2019

This uses the model described in section II of the following paper to estimate
the subshell photoionization quantum efficiency
B. L. Henke, "Ultrasoft-X-Ray Reflection, Refraction, and Production of 
Photoelectrons (100-1000 eV Region)*" 
Phys. Rev. A Volume 6, Number 1 p. 94-104 (1972)

@author: dhigley
"""

import numpy as np
import matplotlib.pyplot as plt

files_start = "/Users/dhigley/Documents/pax_simulations_manuscript/manuscript_plots/conversion_efficiency/"

# binding energies taken from CXRO:
ag_3d_3half_binding = 374.0
ag_3d_5half_binding = 368.3
au_4f_5half_binding = 87.6
au_4f_7half_binding = 84.0
al_2p_1half_bidning = 72.95
al_2p_3half_binding = 72.55
# Pt Fermi taken as Pt 5d value
pt_fermi_binding = 11.4

# Photoionization cross sections from Yeh and Lindau (in Mb, 10^(-22) m^2 = 10^(-18) cm^2):
ag_3d_data = np.genfromtxt("".join([files_start, "ag_3d_cross_sections.txt"]))
ag_3d_cross_section = {"x": ag_3d_data[:, 0], "y": ag_3d_data[:, 1]}
au_4f_data = np.genfromtxt("".join([files_start, "au_4f_cross_sections.txt"]))
au_4f_cross_section = {"x": au_4f_data[:, 0], "y": au_4f_data[:, 1]}
al_2p_data = np.genfromtxt("".join([files_start, "al_2p_cross_sections.txt"]))
al_2p_cross_section = {"x": al_2p_data[:, 0], "y": al_2p_data[:, 1]}
# (use Pt 5d cross section)
pt_5d_data = np.genfromtxt("".join([files_start, "pt_5d_cross_sections.txt"]))
pt_fermi_cross_section = {"x": pt_5d_data[:, 0], "y": pt_5d_data[:, 1]}
# IMFPs are in nm
# IMFP for Ag and Au from S. Tanuma et al.,
# "Experimental determinations of electron inelastic mean free paths in
#  silver, gold, copper and silicon from electron elastic peak intensity
#  ratios" Journal of Surface Analysis volume 9, number 3 (2002)
# (IMFP is in Angstroms)
# IMFP for Al, Pt from H. Shinotsuka et al.,
# "Calculations of electron inelastic mean free paths. X. Data for 41 elemental
#  solids over the 50 eV to 200 keV range with the relativistic full Penn
#  algorithm" Surf. Interface Anal. 2015, 47, 871-888 (2015)
ag_imfp = {
    "x": np.array(
        [
            50,
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
        ]
    ),
    "y": np.array(
        [
            5.3,
            4.3,
            5.3,
            6.5,
            7.7,
            8.9,
            10.0,
            11.1,
            12.1,
            13.2,
            14.2,
            15.2,
            16.2,
            17.1,
            18.1,
            19.0,
        ]
    ),
}
au_imfp = {
    "x": np.array(
        [
            50,
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
        ]
    ),
    "y": np.array(
        [
            3.6,
            3.5,
            4.6,
            5.7,
            6.8,
            7.9,
            8.9,
            9.9,
            10.9,
            11.8,
            12.8,
            13.7,
            14.6,
            15.5,
            16.3,
            17.2,
        ]
    ),
}
al_imfp = {
    "x": np.array(
        [
            54.6,
            60.3,
            66.7,
            73.7,
            81.5,
            90.0,
            99.5,
            109.9,
            121.5,
            134.3,
            148.4,
            164.0,
            181.3,
            200.3,
            221.4,
            244.7,
            270.4,
            298.9,
            330.3,
            365.0,
            403.4,
            445.9,
            492.7,
            544.6,
            601.8,
            665.1,
            735.1,
            812.4,
            897.8,
            992.3,
            1096.6,
            1212.0,
            1339.4,
            1480.3,
            1636.0,
            1808,
        ]
    ),
    "y": np.array(
        [
            0.357,
            0.368,
            0.381,
            0.397,
            0.415,
            0.435,
            0.458,
            0.483,
            0.511,
            0.541,
            0.575,
            0.611,
            0.651,
            0.693,
            0.739,
            0.788,
            0.841,
            0.899,
            0.966,
            1.03,
            1.10,
            1.18,
            1.27,
            1.36,
            1.46,
            1.57,
            1.69,
            1.83,
            1.97,
            2.13,
            2.30,
            2.49,
            2.69,
            2.91,
            3.15,
            3.42,
        ]
    ),
}
pt_imfp = {
    "x": np.array(
        [
            54.6,
            60.3,
            66.7,
            73.7,
            81.5,
            90.0,
            99.5,
            109.9,
            121.5,
            134.3,
            148.4,
            164.0,
            181.3,
            200.3,
            221.4,
            244.7,
            270.4,
            298.9,
            330.3,
            365.0,
            403.4,
            445.9,
            492.7,
            544.6,
            601.8,
            665.1,
            735.1,
            812.4,
            897.8,
            992.3,
            1096.6,
            1212.0,
            1339.4,
            1480.3,
            1636.0,
            1808,
        ]
    ),
    "y": np.array(
        [
            0.501,
            0.483,
            0.467,
            0.453,
            0.442,
            0.436,
            0.435,
            0.437,
            0.442,
            0.450,
            0.460,
            0.472,
            0.488,
            0.506,
            0.527,
            0.550,
            0.577,
            0.608,
            0.642,
            0.680,
            0.722,
            0.768,
            0.818,
            0.872,
            0.931,
            0.996,
            1.07,
            1.14,
            1.22,
            1.31,
            1.41,
            1.52,
            1.63,
            1.76,
            1.89,
            2.04,
        ]
    ),
}

avogadro_number = 6.022e23
# densities in units of g/cm^3
al_density = 2.7
ag_density = 10.49
au_density = 19.32
pt_density = 21.45
# molar mass are in units of g/mol
al_molar_mass = 26.981539
ag_molar_mass = 107.8682
au_molar_mass = 196.96657
pt_molar_mass = 195.084
# atomic density (number/cm^3)
al_atomic_density = avogadro_number * al_density / al_molar_mass
ag_atomic_density = avogadro_number * ag_density / ag_molar_mass
au_atomic_density = avogadro_number * au_density / au_molar_mass
pt_atomic_density = avogadro_number * pt_density / pt_molar_mass


def conversion_plot():
    phot = np.linspace(500, 1500, 1000)
    ag_conversion_efficiency = {
        "x": phot,
        "y": calculate_conversion_efficiency(
            phot, ag_3d_cross_section, ag_imfp, ag_3d_5half_binding, ag_atomic_density
        ),
    }
    phot = np.linspace(200, 1500, 1000)
    au_conversion_efficiency = {
        "x": phot,
        "y": calculate_conversion_efficiency(
            phot, au_4f_cross_section, au_imfp, au_4f_7half_binding, au_atomic_density
        ),
    }
    al_conversion_efficiency = {
        "x": phot,
        "y": calculate_conversion_efficiency(
            phot, al_2p_cross_section, al_imfp, al_2p_3half_binding, al_atomic_density
        ),
    }
    pt_conversion_efficiency = {
        "x": phot,
        "y": calculate_conversion_efficiency(
            phot, pt_fermi_cross_section, pt_imfp, pt_fermi_binding, pt_atomic_density
        ),
    }
    plt.figure(figsize=(3.37, 3.5))
    plt.semilogy(
        ag_conversion_efficiency["x"], ag_conversion_efficiency["y"], label="Ag 3d"
    )
    plt.semilogy(
        au_conversion_efficiency["x"], au_conversion_efficiency["y"], label="Au 4f"
    )
    plt.semilogy(
        al_conversion_efficiency["x"], al_conversion_efficiency["y"], label="Al 2p"
    )
    plt.semilogy(
        pt_conversion_efficiency["x"], pt_conversion_efficiency["y"], label="Pt Fermi"
    )
    plt.legend(loc="center right")
    plt.xlabel("Photon Energy (eV)")
    plt.ylabel("Conversion Efficiency\n(Electrons per Photon)")
    plt.xlim((250, 1500))
    plt.tight_layout()
    plt.savefig("figures/2020_09_28_conversion_efficiency.eps", dpi=600)


def calculate_conversion_efficiency(
    phot, cross_section, imfp, binding_energy, number_density
):
    kinetic_energy = phot - binding_energy
    cross_interp = np.interp(phot, cross_section["x"], cross_section["y"])
    imfp_interp = np.interp(kinetic_energy, imfp["x"], imfp["y"])
    conversion_efficiency = (
        number_density * (cross_interp * 1e-18) * (imfp_interp / 1e7)
    )
    return conversion_efficiency
