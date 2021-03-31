#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: bootstrap_sample.py
"""
Created on Mon Mar 29 15:55:07 2021

@author: Neo(niu.liu@nju.edu.cn)

Use the nonparameter bootstrap resampling to estimate the confidence level of
VSH parameters.

"""

import numpy as np

from .auto_elim import extract_data
from .matrix_calc import nor_eq_sol
from .pmt_convert import convert_ts_to_rgq


def bs_freq(X=None):
    """Determine bootstrap resampling frequency for sample X

    N ~ n * ln(n)^2, n = len(X)

    Reference:
    Feigelson, E., & Babu, G. (2012). Modern Statistical Methods for Astronomy:
    With R Applications. Cambridge: Cambridge University Press. P 55.
    doi:10.1017/CBO9781139015653

    Parameter
    ---------
    X : array_like, optional
        sample to be resampled

    Return
    ------
    N : int
        resampling frequency
    """

    print("Decide the frequency of bootstrap resamplings")
    if X == None:
        print("Since no sample is given, I will use the default value (1000)")
        N = 1000

    else:
        n = len(X)
        N = int(n * np.log(n)**2)

        print("Sample size n={}, as a result, the resampling frequency is "
              "N ~ n * ln(n)^2 = {}".format(n, N))

    return N


def bs_formal_error(x, xi):
    """Estimate formal error (Std)

    Parameters
    ----------
    x : float
        estimation
    xi : array-like, float
        estimations from bootstrap resampling

    Returns
    -------
    xi_mean : float
        mean value of xi
    xi_std : float
        standard deviation of xi
    x_std : float
        RMS of xi with respect to x

    """

    xi_mean = np.mean(xi)
    xi_std = np.std(xi)
    xi_rms = np.sqrt(np.sum((xi-x)**2)/(len(xi)-1))
    # Another way to estimate the confident interval is, e.g.,
    # np.percentile(xi, [2.5, 97.5])

    return xi_mean, xi_std, xi_rms


def bs_resampling(X, Y, samp_size=None):
    """Return a new sample from bootstrap resampling

    Parameters
    ---------
    X, Y : array-like, float
from .matrix_calc import nor_eq_sol, residual_calc
        sample to be resampled
    samp_size : array-like, optional
        size of the new sample

    Returns
    ------
    new_X, new_Y : array-like, float
        new samples
    """

    if samp_size is None:
        samp_size = len(X)

    indice = np.arange(len(X))

    new_samp_indx = np.random.choice(indice, size=samp_size)
    new_X, new_Y = X[new_samp_indx], Y[new_samp_indx]

    return new_X, new_Y


def bs_resampling_indx(X, samp_size=None):
    """Return a new sample from bootstrap resampling

    Parameters
    ---------
    X : array-like, float
        sample to be resampled
    samp_size : array-like, optional
        size of the new sample

    Returns
    ------
    new_samp_indx : array-like, float
        indice for the new samples
    """

    if samp_size is None:
        samp_size = len(X)

    indice = np.arange(len(X))

    new_samp_indx = np.random.choice(indice, size=samp_size)

    return new_samp_indx


def bootstrap_resample_4_err(mask, dra, ddc, dra_err, ddc_err, ra_rad, dc_rad,
                             ra_dc_cor, l_max, fit_type, num_iter, output):
    """Estimate formal uncertainty using bootstrap resampling technique

    Parameters
    ----------
    mask : array-like of boolean
        flag to find the clean sample
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences in dex
    dra_err/ddc_err : array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc in dex
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
    num_iter : int
        number of source once processed. 100 should be fine
    output : dict
        VSH LSQ fitting output information

    Return
    ------
    output : dict
        add new formal uncertainty estimate

    """

    # Retrieve the clean sample
    if not np.all(mask):
        dra, ddc, dra_err, ddc_err, ra_rad, dc_rad, ra_dc_cor = extract_data(
            mask, dra, ddc, dra_err, ddc_err, ra_rad, dc_rad, ra_dc_cor)

    # Retrieve estimates from the clean sample
    pmt = output["pmt"]
    if "pmt1" in output.keys():
        pmt1 = output["pmt1"]
    else:
        pmt1 = output["pmt2"]

    # Re-sampling and parameter estimation
    N_samp = len(dra)
    N_resamp = bs_freq(N_samp)
    N_pmt = len(pmt)
    N_pmt1 = len(pmt1)
    pmt_ts_array = np.zeros((N_resamp, N_pmt))  # t_lm/s_lm coefficient
    # rotation/glide/quadrupolar term
    pmt_rgq_array = np.zeros((N_resamp, N_pmt1))

    for i in range(N_resamp):
        # Generate a new sample
        new_indx = bs_resampling_indx(dra)
        dra1, ddc1, dra_err1, ddc_err1, ra_rad1, dc_rad1, ra_dc_cor1 = extract_data(
            new_indx, dra, ddc, dra_err, ddc_err, ra_rad, dc_rad, ra_dc_cor)

        # estimate t_lm/s_lm coefficients from the new sample
        pmti, sigi, cor_mati, dra_ri, ddc_ri = nor_eq_sol(
            drai, ddci, dra_erri, ddc_erri, ra_radi, dc_radi, ra_dc_cor=ra_dc_cori,
            l_max=l_max, fit_type=fit_type, num_iter=num_iter)
        pmt_ts_array[i, :] = pmti

        # Convert to rotation/glide/quadrupolar terms
        pmt1i, err1i, cor_mat1i = convert_ts_to_rgq(
            pmti, sigi, cor_mati, l_max, fit_type)
        pmt_rgq_array[i, :] = pmt1i

    # Delete several unused varaibles to free space
    del dra1, ddc1, dra_err1, ddc_err1, ra_rad1, dc_rad1, ra_dc_cor1

    # Estimate the formal uncertainty
    # For t_lm/s_lm cpefficient
    pmt_ts_mean = np.zeros(N_pmt)
    pmt_ts_std = np.zeros(N_pmt)
    pmt_ts_rms = np.zeros(N_pmt)

    for i in range(N_pmt):
        meani, stdi, rmsi = bs_formal_error(pmt[i], pmt_ts_array[:, i])
        pmt_ts_mean[i] = meani
        pmt_ts_std[i] = stdi
        pmt_ts_rms[i] = rmsi

    # For rotation/glide/quadrupolar terms
    pmt_rgq_mean = np.zeros(N_pmt1)
    pmt_rgq_std = np.zeros(N_pmt1)
    pmt_rgq_rms = np.zeros(N_pmt1)

    for i in range(N_pmt1):
        meani, stdi, rmsi = bs_formal_error(pmt1[i], pmt_rgq_array[:, i])
        pmt_rgq_mean[i] = meani
        pmt_rgq_std[i] = stdi
        pmt_rgq_rms[i] = rmsi

    # Store the results
    output["pmt_bs_mean"] = pmt_ts_mean
    output["pmt_bs_std"] = pmt_ts_std
    output["pmt_bs_rms"] = pmt_ts_rms

    if "pmt1" in output.keys():
        output["pmt1_bs_mean"] = pmt_rgq_mean
        output["pmt1_bs_std"] = pmt_rgq_std
        output["pmt1_bs_rms"] = pmt_rgq_rms
    else:
        output["pmt2_bs_mean"] = pmt_rgq_mean
        output["pmt2_bs_std"] = pmt_rgq_std
        output["pmt2_bs_rms"] = pmt_rgq_rms

    # Add note
    output["note"] = output["note"] + [
        "pmt_bs_mean: VSH coeffiecent mean from bootstrap sampling\n"
        "pmt_bs_std: VSH coefficient std from bootstrap sampling\n"
        "pmt_bs_rms: VSH coefficient rms from bootstrap sampling\n"
        "pmt1(2)_bs_mean: glide+rotation(+quadrupolar) mean from bootstrap sampling\n"
        "pmt1(2)_bs_std: glide+rotation(+quadrupolar) std from bootstrap sampling\n"
        "pmt1(2)_bs_rms: glide+rotation(+quadrupolar) rms from bootstrap sampling\n"][0]

    return output


def main():
    """Maybe add some tests here
    """

    print("Have a nice day!")


if __name__ == "__main__":
    main()
# --------------------------------- END --------------------------------
