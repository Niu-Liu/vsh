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
from matrix_calc_201128 import nor_eq_sol
from vsh_significance import check_vsh_sig
from vsh_aux_info import prefit_check, prefit_info
from vsh_stat import print_stats_info, residual_statistics_calc
from pmt_convert import convert_ts_to_rotgli


# -----------------------------  MAIN -----------------------------
def vsh_fit(dra, ddc, dra_err, ddc_err, ra, dc, ra_dc_cov=None,
            ra_dc_cor=None, l_max=1, fit_type="full", pos_in_rad=False,
            num_iter=100):
    """
    Parameters
    ----------
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences in dex
    dra_err/ddc_err : array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc in dex
    ra_rad/dc_rad : array of float
        Right ascension/Declination in radian
    ra_dc_cov/ra_dc_cor : array of float
        covariance/correlation coefficient between dra and ddc in dex^2, default is None
    l_max : int
        maximum degree
    fit_type : string
        flag to determine which parameters to be fitted
        "full" for T- and S-vectors both
        "T" for T-vectors only
        "S" for S-vectors only
    pos_in_rad : Boolean
        tell if positions are given in radian, mostly False
    num_iter : int
        number of source once processed. 100 should be fine

    Returns
    ----------
    pmt : array of float
        estimation of (d1, d2, d3, r1, r2, r3) in dex
    sig : array of float
        uncertainty of x in dex
    cor_mat : matrix
        matrix of correlation coefficient.
    """

    # Step 1: pre-fit preparation
    #
    if pos_in_rad:
        ra_rad, dc_rad = ra, dc
    else:
        # Degree -> radian
        ra_rad = np.deg2rad(ra)
        dc_rad = np.deg2rad(dc)

    # # 1.1 check if conditions for fit are all met
    # # Number of sources
    # num_sou = len(dra)
    #
    # # Number of unknowns to deternine
    # num_pmt = 2 * l_max * (l_max + 2)
    #
    # # Pre-fit check
    # prefit_check(num_sou, num_pmt)
    #
    # # Number of degree of freedom
    # num_dof = 2 * num_sou - num_pmt - 1
    #
    # # 1.2 Print basic information
    # print("---------------------- VSH Fit (by Niu LIU) ----------------------")
    # prefit_info(num_sou, num_pmt, num_dof, l_max)

    # Step 2: Fitting
    # Add some loops to detect and eliminate outliers automately?
    # Do the fitting
    # pmt, sig, cor_mat, dra1, ddc1 = nor_eq_sol(dra, ddc, dra_err, ddc_err, ra_rad,
    #                                            dc_rad, ra_dc_cov, ra_dc_cor,
    #                                            l_max, fit_type, num_iter)
    # Calculate residual of observation equations

    # Step 3: Calculate statistics of data
    # for pre-fit data
    # apr_statis = residual_statistics_calc(dra, ddc, dra_err, ddc_err, ra_dc_cov,
    #                                       ra_dc_cor, num_dof)
    #
    # # for post-fit data
    # pos_statis = residual_statistics_calc(dra1, ddc1, dra_err, ddc_err, ra_dc_cov,
    #                                       ra_dc_cor, num_dof)
    #
    # # Print statistics information
    # print_stats_info(apr_statis, pos_statis)
    #
    # # Step 4: Check the significance level
    # check_vsh_sig(dra, ddc, pmt, sig, l_max)
    #
    # # Step 5: post-fit treatments
    # # Rescale the formal errors
    # sig = sig * np.sqrt(pos_statis["reduced_chi2"])
    #
    # # First degrees -> rotation/glide
    # convert_ts_to_rotgli(pmt, sig, cor_mat)
    pmt, sig, cor_mat = nor_eq_sol(dra, ddc, dra_err, ddc_err, ra_rad,
                                   dc_rad, ra_dc_cov, ra_dc_cor,
                                   l_max, fit_type, num_iter)
    return pmt, sig, cor_mat


def vsh_fit_4_Table(data_tab, l_max=1, fit_type="full", pos_in_rad=False,
                    num_iter=100):
    """VSH fit for Atstropy.Table

    Parameters
    ----------
    data_tab : Astropy.table-like
        must contain column names of ["dra", "ddec", "ra", "dec",
        "dra_err", "ddec_err"]
    l_max : int
        maximum degree
    fit_type : string
        flag to determine which parameters to be fitted
        "full" for T- and S-vectors both
        "T" for T-vectors only
        "S" for S-vectors only
    pos_in_rad : Boolean
        tell if positions are given in radian, mostly False
    num_iter : int
        number of source once processed. 100 should be fine

    Returns
    ----------
    pmt : array of float
        estimation of (d1, d2, d3, r1, r2, r3) in dex
    sig : array of float
        uncertainty of x in dex
    cor_mat : matrix
        matrix of correlation coefficient.
    """
    # Transform astropy.Column into np.array
    dra = np.array(data_tab["dra"])
    ddec = np.array(data_tab["ddec"])
    ra = np.array(data_tab["ra"])
    dec = np.array(data_tab["dec"])
    dra_err = np.array(data_tab["dra_err"])
    ddec_err = np.array(data_tab["ddec_err"])

    if "dra_ddec_cov" in data_tab.colnames:
        dra_ddc_cov = np.array(data_tab["dra_ddec_cov"])
        dra_ddc_cor = None
    elif "dra_ddec_cor" in data_tab.colnames:
        dra_ddc_cov = None
        dra_ddc_cor = np.array(data_tab["dra_ddec_cor"])
    else:
        dra_ddc_cov = None
        dra_ddc_cor = None

    # DO the LSQ fitting
    pmt, err, cor_mat = vsh_fit(dra, ddec, dra_err, ddec_err, ra, dec,
                                dra_ddc_cov, dra_ddc_cor,
                                l_max, pos_in_rad, num_iter)

    return pmt, err, cor_mat
# --------------------------------- END --------------------------------
