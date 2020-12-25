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

import sys
import os
import numpy as np
from numpy import pi, concatenate

# My progs
# from .vsh_expension import real_vec_sph_harm_proj
from vsh_expension_201125 import real_vec_sph_harm_proj


# -----------------------------  FUNCTIONS -----------------------------
def cov_mat_calc(dra_err, ddc_err, ra_dc_cov=None, ra_dc_cor=None):
    """Calculate the covariance matrix.

    Parameters
    ----------
    dra_err/ddc_err : float
        formal uncertainty of dRA(*cos(dc_rad))/dDE
    ra_dc_cov : float
        correlation between dRA and dDE, default is None
    ra_dc_cor : float
        correlation coefficient between dRA and dDE, default is None

    Returns
    ----------
    cov_mat : matrix
        Covariance matrix used in the least squares fitting.
    """

    # Covariance matrix.
    cov_mat = np.array([[dra_err**2, 0], [0, ddc_err**2]])

    # Take the correlation into consideration.
    # Assume at most one of ra_dc_cov and ra_dc_cor is given
    if ra_dc_cov is None and ra_dc_cor is not None:
        ra_dc_cov = ra_dc_cor * dra_err * ddc_err

    if ra_dc_cov is not None:
        cov_mat[0, 1] = ra_dc_cov
        cov_mat[1, 0] = ra_dc_cov

    return cov_mat


def cov_to_cor(cov_mat):
    """Convert covariance matrix to sigma and correlation coefficient matrix
    """

    # Formal uncertainty
    sig = np.sqrt(cov_mat.diagonal())

    # Correlation coefficient.
    cor_mat = np.array([cov_mat[i, j] / sig[i] / sig[j]
                        for j in range(len(sig))
                        for i in range(len(sig))])
    cor_mat.resize((len(sig), len(sig)))

    return sig, cor_mat


def cor_to_cov(sig, cor_mat):
    """Convert correlation coefficient matrix to sigma and covariance matrix
    """

    # Covariance
    cov_mat = np.array([cor_mat[i, j] * sig[i] * sig[j]
                        for j in range(len(sig))
                        for i in range(len(sig))])
    cov_mat.resize((len(sig), len(sig)))

    return cov_mat


def wgt_mat_calc(dra_err, ddc_err, ra_dc_cov=None, ra_dc_cor=None):
    """Calculate the weight matrix.

    Parameters
    ----------
    dra_err/ddc_err : array of float
        formal uncertainty of dRA(*cos(dc_rad))/dDE
    ra_dc_cov : array of float
        correlation between dRA and dDE, default is None
    ra_dc_cor : array of float
        correlation coefficient between dRA and dDE, default is None

    Returns
    ----------
    wgt_matrix : matrix
        weighted matrix used in the least squares fitting.
    """

    # Calculate the covariance matrix
    cov_mat = cov_mat_calc(dra_err, ddc_err, ra_dc_cov, ra_dc_cor)

    # Inverse it to obtain weight matrix.
    wgt_mat = np.linalg.inv(cov_mat)

    return wgt_mat


def jac_mat_calc(ra_rad, dc_rad, l_max, fit_type="full"):
    """Calculate the Jacobian matrix

    Parameters
    ----------
    ra_rad/dc_rad : array (m,) of float
        Right ascension/Declination in radian
    l_max : int
        maximum degree
    fit_type : string
        flag to determine which parameters to be fitted
        "full" for T- and S-vectors both
        "T" for T-vectors only
        "S" for S-vectors only

    Returns
    ----------
    jac_mat : array of float
        Jacobian matrix  (M, N) (assume N unknows to determine)
    """

    Tlmr_ra, Tlmr_dc, Tlmi_ra, Tlmi_dc = real_vec_sph_harm_proj(
        l_max, ra_rad, dc_rad)

    # Note the relation between Tlm and Slm
    #    Slmr_ra, Slmi_ra = -Tlmr_dc, -Tlmi_dc
    #    Slmr_dc, Slmi_dc = Tlmr_ra, Tlmi_ra

    # Partial
    par_ra = concatenate((Tlmr_ra, Tlmi_ra, -Tlmr_dc, -Tlmi_dc))
    par_dc = concatenate((Tlmr_dc, Tlmi_dc, Tlmr_ra, Tlmi_ra))

    # Treat (ra, dc) as two observations (dependent or independent)
    # Combine the Jacobian matrix projection on RA and Decl.
    len_par = len(par_ra)
    jac_mat = concatenate((par_ra.reshape(1, len_par),
                           par_dc.reshape(1, len_par)), axis=0)

    # Do not forget to remove the imaginary part of m=0 term
    num1 = int((l_max + 3) * l_max / 2)
    ind = []

    for m in range(1, l_max+1):
        num2 = int((m+1)*m/2-1)

        # The real part of m=0 terms should not be multiplied by 2
        jac_mat[:, num2] = jac_mat[:, num2] / 2
        jac_mat[:, num1*2+num2] = jac_mat[:, num1*2+num2] / 2

        # Remove imaginmary part of m=0 terms
        ind.append(num1+num2)
        ind.append(num1*3+num2)

    jac_mat = np.delete(jac_mat, ind, axis=1)

    # Check the shape of the Jacobian matrix
    N = 2 * l_max * (l_max+2)
    if jac_mat.shape != (2, N):
        print("Shape of Jocabian matrix is ({},{}) "
              "rather than ({},{})".format(jac_mat.shape[0], jac_mat.shape[1],
                                           2, N))
        sys.exit()

    return jac_mat


def nor_mat_calc(dra, ddc, dra_err, ddc_err, ra_rad, dc_rad,
                 ra_dc_cov=None, ra_dc_cor=None, l_max=1, fit_type="full"):
    """Cacluate the normal and right-hand-side matrix for LSQ analysis.
    Parameters
    ----------
    dra_err/ddc_err : float
        formal uncertainty of dRA(*cos(dc_rad))/dDE in uas
    ra_rad/dc_rad : float
        Right ascension/Declination in radian
    ra_dc_cov : float
        correlation between dRA and dDE, default is None
    ra_dc_cor : float
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
    nor_mat : array of float
        normal matrix
    rhs_mat : array of float
        right-hand-side matrix
    """

    # Jacobian matrix
    jac_mat = jac_mat_calc(ra_rad, dc_rad, l_max, fit_type)

    # Weighted matrix
    wgt_mat = wgt_mat_calc(dra_err, ddc_err, ra_dc_cov, ra_dc_cor)

    # Jac_mat_T * Wgt_mat
    mul_mat = np.dot(np.transpose(jac_mat), wgt_mat)

    # Calculate normal matrix A
    nor_mat = np.dot(mul_mat, jac_mat)

    # Calculate right-hand-side matrix b
    res_mat = np.array([dra, ddc])
    rhs_mat = np.dot(mul_mat, res_mat)

    return nor_mat, rhs_mat


def nor_eq_sol(dra, ddc, dra_err, ddc_err, ra_rad, dc_rad, ra_dc_cov=None,
               ra_dc_cor=None, l_max=1, fit_type="full"):
    """The 1st degree of VSH function: glide and rotation.
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
    fit_type : string
        flag to determine which parameters to be fitted
        "full" for T- and S-vectors both
        "T" for T-vectors only
        "S" for S-vectors only
    Returns
    ----------
    pmt : array of float
        estimaation of (d1, d2, d3, r1, r2, r3) in uas
    sig : array of float
        uncertainty of x in uas
    cor_mat : matrix
        matrix of correlation coefficient.
    """

    A, b = 0, 0

    for i in range(len(dra)):
        if not ra_dc_cov is None:
            An, bn = nor_mat_calc(dra[i], ddc[i], dra_err[i], ddc_err[i],
                                  ra_rad[i], dc_rad[i], ra_dc_cov=ra_dc_cov[i],
                                  l_max=l_max, fit_type=fit_type)
        elif not ra_dc_cor is None:
            An, bn = nor_mat_calc(dra[i], ddc[i], dra_err[i], ddc_err[i],
                                  ra_rad[i], dc_rad[i], ra_dc_cor=ra_dc_cor[i],
                                  l_max=l_max, fit_type=fit_type)
        else:
            An, bn = nor_mat_calc(dra[i], ddc[i], dra_err[i], ddc_err[i],
                                  ra_rad[i], dc_rad[i],
                                  l_max=l_max, fit_type=fit_type)
        A = A + An
        b = b + bn

    # Solve the equations.
    pmt = np.linalg.solve(A, b)

    # Covariance.
    cov_mat = np.linalg.inv(A)
    # Formal uncertainty and correlation coefficient
    sig, cor_mat = cov_to_cor(cov_mat)

    # Return the result.
    return pmt, sig, cor_mat

# --------------------------------- END --------------------------------
