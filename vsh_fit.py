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
from .matrix_calc import nor_eq_sol, calc_residual
from .vsh_significance import check_vsh_sig
from .vsh_aux_info import prefit_check, prefit_info
from .vsh_stats import print_stats_info, residual_stats_calc
from .auto_elim import auto_elim_vsh_fit
from .pmt_convert import convert_ts_to_rotgli, st_to_rotgld, st_to_rotgldquad


# -----------------------------  MAIN -----------------------------
def vsh_fit(dra, ddc, dra_err, ddc_err, ra, dc, ra_dc_cor=None,
            l_max=1, fit_type="full", pos_in_rad=False,
            num_iter=100, clip_limit=None):
    """VSH fitted to an offset field

    Parameters
    ----------
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    dra_err/ddc_err : array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc in uas
    ra_rad/dc_rad : array of float
        Right ascension/Declination in radian
    ra_dc_cov/ra_dc_cor : array of float
        covariance/correlation coefficient between dra and ddc in uas^2, default is None
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
    clip_limit : float
        thershold on normalized separation for auto-elimination

    Returns
    ----------
    pmt : array of float
        estimation of (d1, d2, d3, r1, r2, r3) in uas
    sig : array of float
        uncertainty of x in uas
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

    # 1.1 check if conditions for fit are all met
    # Number of sources
    num_sou = len(dra)

    # Number of unknowns to deternine
    num_pmt = 2 * l_max * (l_max + 2)

    # Pre-fit check
    prefit_check(num_sou, num_pmt)

    # Number of degree of freedom
    num_dof = 2 * num_sou - num_pmt - 1

    # 1.2 Print basic information
    print("---------------------- VSH Fit (by Niu LIU) ----------------------")
    prefit_info(num_sou, num_pmt, num_dof, l_max)

    # Step 2: Fitting
    if clip_limit is None:
        print("No constraints put on the data, so the fitting will be done only once.")
        # Do the fitting
        pmt, sig, cor_mat, dra_r, ddc_r = nor_eq_sol(
            dra, ddc, dra_err, ddc_err, ra_rad, dc_rad, ra_dc_cor=ra_dc_cor,
            l_max=l_max, fit_type=fit_type, num_iter=num_iter)

    elif isinstance(clip_limit, (int, float)):
        print("The fit will be iterated until it converges, "
              "that is, no more outliers, based on a clean sample of "
              "a normalized separation X<={:.3f}*median(X).\n".format(clip_limit))

        # Fitting with auto-elimination
        pmt, sig, cor_mat, dra_r, ddc_r = auto_elim_vsh_fit(
            clip_limit, dra, ddc, dra_err, ddc_err, ra_rad, dc_rad,
            ra_dc_cor=ra_dc_cor, l_max=l_max, fit_type=fit_type, num_iter=num_iter)

    else:
        print("Parameter 'clip_limit' must be a float-like value.")

    # Step 3: Calculate statistics of data
    # for pre-fit data
    apr_stats = residual_stats_calc(
        dra, ddc, dra_err, ddc_err, ra_dc_cor, num_dof)

    # for post-fit data
    pos_stats = residual_stats_calc(
        dra_r, ddc_r, dra_err, ddc_err, ra_dc_cor, num_dof)

    # Print statistics information
    print_stats_info(apr_stats, pos_stats)

    # Step 4: Check the significance level
    check_vsh_sig(dra, ddc, pmt, sig, l_max)

    # Step 5: post-fit treatments
    # Rescale the formal errors
    sig = sig * np.sqrt(pos_stats["reduced_chi2"])

    # First degrees -> rotation/glide
    convert_ts_to_rotgli(pmt, sig, cor_mat)

    return pmt, sig, cor_mat


def vsh_fit_4_Table(data_tab, l_max=1, fit_type="full", pos_in_rad=False,
                    num_iter=100, clip_limit=None):
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
        estimation of (d1, d2, d3, r1, r2, r3) in uas
    sig : array of float
        uncertainty of x in uas
    cor_mat : matrix
        matrix of correlation coefficient.
    """

    # Transform astropy.Column into np.array
    if "dra" in data_tab.colnames:
        dra = np.array(data_tab["dra"])
    elif "pmra" in data_tab.colnames:
        dra = np.array(data_tab["pmra"])
    else:
        print(""dra" or "pmra" is not specificed.")
        exit(1)

    if "ddec" in data_tab.colnames:
        ddec = np.array(data_tab["dra"])
    elif "pmdec" in data_tab.colnames:
        ddec = np.array(data_tab["pmdec"])
    else:
        print(""ddec" or "pmdec" is not specificed.")
        exit(1)

    ra = np.array(data_tab["ra"])
    dec = np.array(data_tab["dec"])

    if "dra_err" in data_tab.colnames:
        dra_err = np.array(data_tab["dra_err"])
    elif "dra_error" in data_tab.colnames:
        dra_err = np.array(data_tab["dra_error"])
    if "pmra_err" in data_tab.colnames:
        dra_err = np.array(data_tab["pmra_err"])
    elif "pmra_error" in data_tab.colnames:
        dra_err = np.array(data_tab["pmra_error"])
    else:
        print("'dra_err', 'dra_error', 'pmra_err' or 'pmra_error' is not specificed.")
        exit(1)

    if "ddec_err" in data_tab.colnames:
        ddec_err = np.array(data_tab["ddec_err"])
    elif "ddec_error" in data_tab.colnames:
        ddec_err = np.array(data_tab["ddec_error"])
    if "pmdec_err" in data_tab.colnames:
        ddec_err = np.array(data_tab["pmdec_err"])
    elif "pmdec_error" in data_tab.colnames:
        ddec_err = np.array(data_tab["pmdec_error"])
    else:
        print("'ddec_err', 'ddec_error', 'pmdec_err' or 'pmdec_error' is not specificed.")
        exit(1)

    if "dra_ddec_cov" in data_tab.colnames:
        dra_ddc_cov = np.array(data_tab["dra_ddec_cov"])
        dra_ddc_cor = dra_ddc_cov / dra_err / ddec_err
    elif "dra_ddec_cor" in data_tab.colnames:
        dra_ddc_cor = np.array(data_tab["dra_ddec_cor"])
    elif "pmra_pmdec_corr" in data_tab.colnames:
        dra_ddc_cor = np.array(data_tab["pmra_pmdec_corr"])
    else:
        dra_ddc_cor = None

    # DO the LSQ fitting
    pmt, err, cor_mat = vsh_fit(dra, ddec, dra_err, ddec_err, ra, dec,
                                ra_dc_cor=dra_ddc_cor, l_max=l_max,
                                pos_in_rad=pos_in_rad, num_iter=num_iter,
                                clip_limit=clip_limit)

    return pmt, err, cor_mat


def rotgli_fit(dra, ddc, dra_err, ddc_err, ra, dc, ra_dc_cor=None,
               l_max=1, fit_type="full", pos_in_rad=False, num_iter=100, clip_limit=None):
    """VSH degree 01 fit and return rotation &glide

    Parameters
    ----------
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    dra_err/ddc_err : array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc in uas
    ra_rad/dc_rad : array of float
        Right ascension/Declination in radian
    ra_dc_cor : array of float
        correlation coefficient between dra and ddc, default is None
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
    pmt1 : array of float
        estimation of (d1, d2, d3, r1, r2, r3) in uas
    sig1 : array of float
        uncertainty of x in uas
    cor_mat1 : matrix
        matrix of correlation coefficient.
    """

    # VSH fit
    pmt, err, cor_mat = vsh_fit(dra, ddc, dra_err, ddc_err, ra, dc,
                                ra_dc_cor=ra_dc_cor, l_max=l_max,
                                fit_type=fit_type, pos_in_rad=pos_in_rad,
                                num_iter=num_iter, clip_limit=clip_limit)

    pmt1, err1, cor_mat1 = st_to_rotgld(pmt[:6], err[:6], cor_mat[:6, :6])

    return pmt1, err1, cor_mat1


def rotgliquad_fit(dra, ddc, dra_err, ddc_err, ra, dc, ra_dc_cor=None,
                   l_max=2, fit_type="full", pos_in_rad=False,
                   num_iter=100, clip_limit=None):
    """VSH degree 02 fit and return rotation, glide, &quadrupolar vectors

    Parameters
    ----------
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    dra_err/ddc_err : array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc in uas
    ra_rad/dc_rad : array of float
        Right ascension/Declination in radian
    ra_dc_cor : array of float
        correlation coefficient between dra and ddc, default is None
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
    pmt2 : array of float
        estimation of (d1, d2, d3, r1, r2, r3,
        E22R, E22I, E21R, E21I, E20, M22R, M22I, M21R, M21I, M20)
    sig2 : array of float
        uncertainty of pmt2
    cor_mat2 : matrix
        matrix of correlation coefficient.
    """

    if l_max < 2:
        print("The maximum degree l_max could not be smaller than 2.")
        sys.exit(1)

    # VSH fit
    pmt, err, cor_mat = vsh_fit(dra, ddc, dra_err, ddc_err, ra, dc, ra_dc_cor=ra_dc_cor,
                                l_max=l_max, fit_type=fit_type, pos_in_rad=pos_in_rad,
                                num_iter=num_iter, clip_limit=clip_limit)

    pmt2, err2, cor_mat2 = st_to_rotgldquad(
        pmt[:16], err[:16], cor_mat[:16, :16])

    return pmt2, err2, cor_mat2


def rotgli_fit_4_Table(data_tab, l_max=1, fit_type="full", pos_in_rad=False,
                       num_iter=100, clip_limit=None):
    """VSH degree 01 fit and return rotation &glide vectors for Atstropy.Table

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
        estimation of (d1, d2, d3, r1, r2, r3) in uas
    sig : array of float
        uncertainty of x in uas
    cor_mat : matrix
        matrix of correlation coefficient.
    """

    # VSH fit
    pmt, err, cor_mat = vsh_fit_4_Table(
        data_tab, l_max=l_max, fit_type=fit_type, pos_in_rad=pos_in_rad,
        num_iter=num_iter, clip_limit=clip_limit)

    pmt1, err1, cor_mat1 = st_to_rotgld(pmt[:6], err[:6], cor_mat[:6, :6])

    return pmt1, err1, cor_mat1


def rotgliquad_fit_4_Table(data_tab, l_max=2, fit_type="full", pos_in_rad=False,
                           num_iter=100, clip_limit=None):
    """VSH degree 02 fit and return rotation, glide, &quadrupolar vectors for Atstropy.Table

    Parameters
    ----------
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    dra_err/ddc_err : array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc in uas
    ra_rad/dc_rad : array of float
        Right ascension/Declination in radian
    ra_dc_cor : array of float
        correlation coefficient between dra and ddc, default is None
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
    pmt2 : array of float
        estimation of (d1, d2, d3, r1, r2, r3,
        E22R, E22I, E21R, E21I, E20, M22R, M22I, M21R, M21I, M20)
    sig2 : array of float
        uncertainty of pmt2
    cor_mat2 : matrix
        matrix of correlation coefficient.
    """

    if l_max < 2:
        print("The maximum degree l_max could not be smaller than 2.")
        sys.exit(1)

    # VSH fit
    pmt, err, cor_mat = vsh_fit_4_Table(
        data_tab, l_max=l_max, fit_type=fit_type, pos_in_rad=pos_in_rad,
        num_iter=num_iter, clip_limit=clip_limit)

    pmt2, err2, cor_mat2 = st_to_rotgldquad(
        pmt[:16], err[:16], cor_mat[:16, :16])

    return pmt2, err2, cor_mat2


def main():
    """Not implemented yet

    """
    pass


if __name__ == "__main__":
    main()

# --------------------------------- END --------------------------------