#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: matrix_calc.py
"""
Created on Thu May  7 20:56:54 2020

@author: Neo(liuniu@smail.nju.edu.cn)

Some codes for calculating various matrix & array needed in the LSQ preocess.

The normal equation is writtern as
    A * x = b
where A is the normal matrix, x the vector consist of unknowns,
and b the right-hand-side array.

"""

import numpy as np
from numpy import sin, cos, pi, concatenate
import sys


# -----------------------------  FUNCTIONS -----------------------------
def cov_mat_calc(dra_err, ddec_err, ra_dec_cov=None, ra_dec_cor=None):
    """Calculate the covariance matrix.

    Parameters
    ----------
    dra_err/ddec_err : array of float
        formal uncertainty of dRA(*cos(dec_rad))/dDE
    ra_dec_cov : array of float
        correlation between dRA and dDE, default is None
    ra_dec_cor : array of float
        correlation coefficient between dRA and dDE, default is None

    Returns
    ----------
    cov_mat : matrix
        Covariance matrix used in the least squares fitting.
    """

    if len(ddec_err) != len(dra_err):
        print("The length of dra_err and ddec_err should be equal")
        sys.exit()

    # TO-DO
    # Check the length of correlation array
    # else:
    #     if ra_dec_cov is None:

    # Covariance matrix.
    err = np.vstack((dra_err, ddec_err)).T.flatten()
    cov_mat = np.diag(err**2)

    # Take the correlation into consideration.
    if ra_dec_cor is not None and ra_dec_cov is None:
        ra_dec_cov = ra_dec_cor * dra_err * ddec_err

    if ra_dec_cov is not None:
        for i, covi in enumerate(ra_dec_cov):
            cov_mat[2*i, 2*i+1] = covi
            cov_mat[2*i+1, 2*i] = covi

    return cov_mat


def wgt_mat_calc(dra_err, ddec_err, ra_dec_cov=None, ra_dec_cor=None):
    """Calculate the weight matrix.

    Parameters
    ----------
    dra_err/ddec_err : array of float
        formal uncertainty of dRA(*cos(dec_rad))/dDE
    ra_dec_cov : array of float
        correlation between dRA and dDE, default is None
    ra_dec_cor : array of float
        correlation coefficient between dRA and dDE, default is None

    Returns
    ----------
    wgt_matrix : matrix
        weighted matrix used in the least squares fitting.
    """

    # Calculate the covariance matrix
    cov_mat = cov_mat_calc(dra_err, ddec_err, ra_dec_cov, ra_dec_cor)

    # Inverse it to obtain weight matrix.
    wgt_mat = np.linalg.inv(cov_mat)

    return wgt_mat

#### TO-DO ####
# Calculate the Jacobian matrix
# Calculate the right-hand-side array
#### TO-DO ####


def nor_mat_calc(dra_err, ddec_err, ra_rad, dec_rad,
                 ra_dec_cov=None, ra_dec_cor=None, l_max=1, fit_type="full"):
    """Cacluate the normal matrix for LSQ analysis.

    Parameters
    ----------
    dra_err/ddec_err : array of float
        formal uncertainty of dRA(*cos(dec_rad))/dDE in uas
    ra_rad/dec_rad : array of float
        Right ascension/Declination in radian
    ra_dec_cov : array of float
        correlation between dRA and dDE, default is None
    ra_dec_cor : array of float
        correlation coefficient between dRA and dDE, default is None
    l_max : int
        maximum degree
    fit_type : string
        flag to determine which parameters to be fitted
        "full" for T- and S-vectors both
        "T" for T-vectors only
        "S" for S-vectors only

    Returns
    ----------
    A : array of float
        normal matrix
    """

    # Jacobian matrix and its transpose.
    jac_mat, jac_mat_T = jac_mat_calc(ra_rad, dec_rad, l_max, fit_type)

    # Weighted matrix.
    wgt_mat = wgt_mat_calc(dra_err, ddec_err, ra_dec_cov, ra_dec_cor)

    # Calculate matrix A:
    A_mat = np.dot(np.dot(jac_mat_T, wgt_mat), jac_mat)

    return A_mat


#### TO-DO ####
# Calculate the right-hand-side array b
#### TO-DO ####
# --------------------------------- END --------------------------------
