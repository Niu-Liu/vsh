#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: auto_elim.py
"""
Created on Thu Dec 24 10:35:28 2020

@author: Neo(niu.liu@nju.edu.cn)

This script contains code for auto-elimination.
"""

import numpy as np

# My progs
from .stats_func import calc_nor_sep
from .matrix_calc import (nor_eq_sol_from_cache, nor_eq_sol,
                          residual_calc_from_cache,
                          cache_mat_calc, rm_cache_mat)


# ----------------------------- Function -----------------------------
def extract_data(mask, dra, ddc, dra_err, ddc_err, ra_rad, dc_rad, ra_dc_cor=None):
    """Get a clean sample based on mask

    Parameters
    ----------
    mask : array of boolean
        mask for extract data
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences
    dra_err/ddc_err : array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc
    ra_rad/dc_rad : array of float
        Right ascension/Declination in radian

    Returns
    ----------
    dra_new/ddc_new: array of float
        R.A.(*cos(Dec.))/Dec for the clean sample. differences
    dra_err_new/ddc_err_new: array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc for the clean sample
    ra_rad_new/dc_rad_new: array of float
        Right ascension/Declination in radian for the clean sample
    ra_dc_cor_new: array of float
        covariance/correlation coefficient between dra and ddc for the clean sample
    """

    # Extract the clean sample
    dra_new, ddc_new = dra[mask], ddc[mask]
    dra_err_new, ddc_err_new = dra_err[mask], ddc_err[mask]
    ra_rad_new, dc_rad_new = ra_rad[mask], dc_rad[mask]

    if ra_dc_cor is None:
        ra_dc_cor_new = ra_dc_cor
    else:
        ra_dc_cor_new = ra_dc_cor[mask]

    return dra_new, ddc_new, dra_err_new, ddc_err_new, ra_rad_new, dc_rad_new, ra_dc_cor_new


def elim_on_nor_sep(clip_limit, dra, ddc, dra_err, ddc_err, ra_dc_cor=None):
    """Get a clean sample based on normalized separation

    Parameters
    ----------
    dra/ddc: array of float
        R.A.(*cos(Dec.))/Dec. differences
    dra_err/ddc_err: array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc
    ra_rad/dc_rad: array of float
        Right ascension/Declination in radian
    clip_limit: int ot float
        maximum normalized separation for clipping data
    ra_dc_cor: array of float
        correlation coefficient between dra and ddc, default is None

    Returns
    ----------
    dra_new/ddc_new: array of float
        R.A.(*cos(Dec.))/Dec for the clean sample. differences
    dra_err_new/ddc_err_new: array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc for the clean sample
    ra_rad_new/dc_rad_new: array of float
        Right ascension/Declination in radian for the clean sample
    ra_dc_cor_new: array of float
        covariance/correlation coefficient between dra and ddc for the clean sample
    """

    # Calculate normalized separation
    nor_sep = calc_nor_sep(dra, dra_err, ddc, ddc_err, ra_dc_cor)
    med_nor_sep = np.median(nor_sep)
    max_nor_sep = clip_limit * med_nor_sep

    # Constraint on normalized separation
    mask = (nor_sep <= max_nor_sep)

    return max_nor_sep, mask


def auto_elim_vsh_fit(clip_limit, dra, ddc, dra_err, ddc_err, ra_rad, dc_rad,
                      ra_dc_cor=None, l_max=1, fit_type="full", num_iter=100):
    """ Fit the VSH parameters with auto-elimination

    Parameters
    ----------
    clip_limit: float
        thershold on normalized separation for auto-elimination
    dra/ddc: array of float
        R.A.(*cos(Dec.))/Dec. differences
    dra_err/ddc_err: array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc
    ra_rad/dc_rad: array of float
        Right ascension/Declination in radian
    ra_dc_cor: array of float
        correlation coefficient between dra and ddc, default is None
    l_max: int
        maximum degree
    fit_type: string
        flag to determine which parameters to be fitted
        full for T - and S-vectors both
        T for T-vectors only
        S for S-vectors only
    pos_in_rad: Boolean
        tell if positions are given in radian, mostly False
    num_iter: int
        number of source once processed. 100 should be fine

    Returns
    ----------
    pmt: array of float
        estimation of(d1, d2, d3, r1, r2, r3)
    sig: array of float
        uncertainty of x
    cor_mat: matrix
        matrix of correlation coefficient.
    """

    print("==================== Auto-elimination ====================")
    print("    Nb_iteration    Nb_sources    Nb_outliers    Threshold")
    print("    {:11d}    {:9d}    {:9d}    {:9.3f}".format(0, len(dra), 0, 0))

    # mask1 = All True
    mask1 = np.full(len(dra), True)
    iter_count = 0

    # first elimination
    max_nor_sep2, mask2 = elim_on_nor_sep(
        clip_limit, dra, ddc, dra_err, ddc_err, ra_dc_cor)

    # Generate cache matrix
    suffix_array = cache_mat_calc(
        dra, ddc, dra_err, ddc_err, ra_rad, dc_rad,
        ra_dc_cor=ra_dc_cor, l_max=l_max, fit_type=fit_type, num_iter=num_iter)

    while np.any(mask1 != mask2):
        # Renew the flags
        max_nor_sep1, mask1 = max_nor_sep2, mask2

        # Generate a clean sample
        [dra1, ddc1, dra_err1, ddc_err1, ra_rad1, dc_rad1, ra_dc_cor1] = extract_data(
            mask1, dra, ddc, dra_err, ddc_err, ra_rad, dc_rad, ra_dc_cor)

        iter_count += 1
        print("    {:11d}    {:9d}    {:9d}    {:9.3f}".format(
            iter_count, len(dra1), len(dra)-len(dra1), max_nor_sep1))

        pmt, sig, cor_mat = nor_eq_sol(
            dra1, ddc1, dra_err1, ddc_err1, ra_rad1, dc_rad1,
            ra_dc_cor=ra_dc_cor1, l_max=l_max, fit_type=fit_type,
            num_iter=num_iter, calc_res=False)

        # Calculate residuals for all sources
        dra_r, ddc_r = residual_calc_from_cache(
            dra, ddc, pmt, suffix_array)

        # Re-eliminate the data
        max_nor_sep2, mask2 = elim_on_nor_sep(
            clip_limit, dra_r, ddc_r, dra_err, ddc_err, ra_dc_cor)

    # If no source is classified as outlier
    if iter_count == 0:
        pmt, sig, cor_mat, dra_r, ddc_r = nor_eq_sol_from_cache(
            dra, ddc, suffix_array, num_iter=num_iter)

    # Remove cache file
    rm_cache_mat(suffix_array)

    return pmt, sig, cor_mat, dra_r, ddc_r


def std_elim_vsh_fit(x_limit, dra, ddc, dra_err, ddc_err, ra_rad, dc_rad,
                     ra_dc_cor=None, l_max=1, fit_type="full", num_iter=100):
    """Fit the VSH parameters with standard-elimination

    Parameters
    ----------
    x_limit: float
        thershold on normalized separation for std-elimination
    dra/ddc: array of float
        R.A.(*cos(Dec.))/Dec. differences
    dra_err/ddc_err: array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc
    ra_rad/dc_rad: array of float
        Right ascension/Declination in radian
    ra_dc_cor: array of float
        correlation coefficient between dra and ddc, default is None
    l_max: int
        maximum degree
    fit_type: string
        flag to determine which parameters to be fitted
        full for T - and S-vectors both
        T for T-vectors only
        S for S-vectors only
    pos_in_rad: Boolean
        tell if positions are given in radian, mostly False
    num_iter: int
        number of source once processed. 100 should be fine

    Returns
    ----------
    pmt: array of float
        estimation of(d1, d2, d3, r1, r2, r3)
    sig: array of float
        uncertainty of x
    cor_mat: matrix
        matrix of correlation coefficient.
    """

    # Generate cache matrix
    suffix_array = cache_mat_calc(
        dra, ddc, dra_err, ddc_err, ra_rad, dc_rad,
        ra_dc_cor=ra_dc_cor, l_max=l_max, fit_type=fit_type, num_iter=num_iter)

    # Calculate normalized separation
    nor_sep = calc_nor_sep(dra, dra_err, ddc, ddc_err, ra_dc_cor)
    mask = (nor_sep <= x_limit)

    # Generate a clean sample
    [dra1, ddc1, dra_err1, ddc_err1, ra_rad1, dc_rad1, ra_dc_cor1] = extract_data(
        mask, dra, ddc, dra_err, ddc_err, ra_rad, dc_rad, ra_dc_cor)

    print("==================== Standard-elimination ====================")
    print("    Nb_sources    Nb_outliers    Threshold")
    print("    {:9d}    {:9d}    {:9.3f}".format(
        len(dra1), len(dra)-len(dra1), x_limit))

    pmt, sig, cor_mat = nor_eq_sol(
        dra1, ddc1, dra_err1, ddc_err1, ra_rad1, dc_rad1,
        ra_dc_cor=ra_dc_cor1, l_max=l_max, fit_type=fit_type,
        num_iter=num_iter, calc_res=False)

    # Calculate residuals for all sources
    dra_r, ddc_r = residual_calc_from_cache(
        dra, ddc, pmt, suffix_array)

    # Remove cache file
    rm_cache_mat(suffix_array)

    return pmt, sig, cor_mat, dra_r, ddc_r


def main():
    """Maybe add some tests here
    """

    print("Have a nice day!")


if __name__ == "__main__":
    main()
# --------------------------------- END --------------------------------
