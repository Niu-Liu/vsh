#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: generate_test_data.py
"""
Created on Tue Nov 24 13:16:49 2020

@author: Neo(niu.liu@nju.edu.cn)


In order to simulate the case of Gaia EDR3 where 1.5M qdexars are included, a
test sample of this amount is generated.
"""

import numpy as np
from astropy.table import Table

# My modules
from .rgq_func import vsh_func


# -----------------------------  MAIN -----------------------------
def generate_test_sample(sample_size, pmt_vec):
    """
    """

    # A sample of sources located as (ra, dec)
    ra = np.random.random_sample(sample_size) * 2 * np.pi
    dec = (np.random.random_sample(sample_size) - 0.5) * np.pi

    # Positional difference from ideal cases
    dra_calc, ddec_calc = vsh_func(ra, dec, pmt_vec, l_max=1)

    # Noise
    dra_nois = 0  # np.random.normal(scale=1, size=sample_size) * 0
    ddec_nois = 0  # np.random.normal(scale=1, size=sample_size) * 0

    # Simulation data
    dra_sim, ddec_sim = dra_calc + dra_nois, ddec_calc + ddec_nois

    # Formal error, uniform one
    dra_err = np.ones(sample_size)
    ddec_err = np.ones(sample_size)
    dra_ddec_cor = np.zeros(sample_size)

    # Generate a Astropy.Table and store the data
    data = Table([ra, dec, dra_sim, ddec_sim, dra_err, ddec_err, dra_ddec_cor],
                 names=["ra", "dec", "dra", "ddec", "dra_err", "ddec_err", "dra_ddec_cor"])

    return data


def main():
    """
    """

    # Read the simulated data
    rot_vec = np.array([20, 30, 15])
    gli_vec = np.array([30, 24, 12])
    pmt_vec = np.concatenate((gli_vec, rot_vec))
    data = generate_test_sample(int(2e6), pmt_vec)
    data.write("test_sample.csv", format="ascii.csv")


if __name__ == "__main__":

    main()
# --------------------------------- END --------------------------------
