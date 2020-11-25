#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: vsh_fit.py
"""
Created on Tue Nov 17 13:48:25 2020

@author: Neo(niu.liu@nju.edu.cn)
"""

import sys
import numpy as np
# My progs
from matrix_calc import nor_eq_sol


# -----------------------------  MAIN -----------------------------
def vsh_fit(dra, ddc, dra_err, ddc_err, ra, dc, ra_dc_cov=None,
            ra_dc_cor=None, l_max=1, fit_type="full", pos_in_rad=False):
    """
    """

    if pos_in_rad:
        ra_rad, dc_rad = ra, dc
    else:
        # Degree -> radian
        ra_rad = np.deg2rad(ra)
        dc_rad = np.deg2rad(dc)

    # TO-DO-1
    # Calculate statistics for pre-fit data

    # Do the fitting
    pmt, sig, cor_mat = nor_eq_sol(dra, ddc, dra_err, ddc_err, ra_rad,
                                   dc_rad, ra_dc_cov, ra_dc_cor, l_max, fit_type)

    # TO-DO-2
    # 2.1 Calculate statistics for post-fit data
    # 2.2 Some loops to detect and eliminate outliers automately?
    # 2.3 Rescale the formal uncertainty

    return pmt, sig, cor_mat
# --------------------------------- END --------------------------------
