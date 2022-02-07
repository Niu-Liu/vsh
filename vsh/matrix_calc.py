#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: matrix_calc.py
'''
Created on Thu May  7 20:56:54 2020

@author: Neo(liuniu@smail.nju.edu.cn)

Some codes for calculating various matrix & array needed in the LSQ preocess.

The normal equation is writtern as
    A * x = b
where A is the normal matrix, x the vector consist of unknowns,
and b the right-hand-side array.

'''

import sys
import os
import numpy as np
from numpy import pi, concatenate
# My progs
# from .vsh_expansion import real_vec_sph_harm_proj
from .vsh_expansion import real_vec_sph_harm_proj, vec_sph_harm_proj

# -----------------------------  FUNCTIONS -----------------------------


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


def cov_mat_calc(dra_err, ddc_err, ra_dc_cor=None):
    """Calculate the covariance matrix.

    Parameters
    ----------
    dra_err/ddc_err : array of float
        formal uncertainty of dRA(*cos(dc_rad))/dDE
    ra_dc_cor : array of float
        correlation coefficient between dRA and dDE, default is None

    Returns
    ----------
    cov_mat : matrix
        Covariance matrix used in the least squares fitting.
    """

    if len(ddc_err) != len(dra_err):
        print('The length of dra_err and ddc_err should be equal')
        sys.exit()

    # Covariance matrix.
    # err = np.vstack((dra_err, ddc_err)).T.flatten()
    err = concatenate((dra_err, ddc_err))
    cov_mat = np.diag(err**2)

    # Take the correlation into consideration.
    # Assume at most one of ra_dc_cov and ra_dc_cor is given
    if ra_dc_cor is not None:
        ra_dc_cov = ra_dc_cor * dra_err * ddc_err

        N = len(dra_err)
        for i, covi in enumerate(ra_dc_cov):
            cov_mat[i, i+N] = covi
            cov_mat[i+N, i] = covi

    return cov_mat


def wgt_mat_calc(dra_err, ddc_err, ra_dc_cor=None):
    """Calculate the weight matrix.

    Parameters
    ----------
    dra_err/ddc_err : array of float
        formal uncertainty of dRA(*cos(dc_rad))/dDE
    ra_dc_cor : array of float
        correlation coefficient between dRA and dDE, default is None

    Returns
    ----------
    wgt_matrix : matrix
        weighted matrix used in the least squares fitting.
    """

    # Calculate the covariance matrix
    cov_mat = cov_mat_calc(dra_err, ddc_err, ra_dc_cor)

    # Inverse it to obtain weight matrix.
    wgt_mat = np.linalg.inv(cov_mat)

    return wgt_mat


def jac_mat_l_calc(T_ra_mat, T_dc_mat, l, fit_type="full"):
    """Calculate the Jacobian matrix of lth degree

    Parameters
    ----------
    ra_rad/dc_rad : array (M,) of float
        Right ascension/Declination in radian
    l : int
        degree of the harmonics
    fit_type : string
        flag to determine which parameters to be fitted
        'full' for T- and S-vectors bot Number of observations
    """

    # Number of source
    M = T_ra_mat.shape[2]

    # Usually begins with the first order
    Tl0_ra, Tl0_dc = real_vec_sph_harm_proj(l, 0, T_ra_mat, T_dc_mat)

    if fit_type in ["full", "FULL", "Full"]:
        # Note the relation between Tlm and Slm
        #    S10_ra, S10_dc = -Tl0_dc, Tl0_ra
        #    jac_mat_ra = concatenate(
        #        (Tl0_ra.reshape(M, 1), Sl0_ra.reshape(M, 1)), axis=1)
        jac_mat_ra = concatenate(
            (Tl0_ra.reshape(M, 1), -Tl0_dc.reshape(M, 1)), axis=1)
        jac_mat_dc = concatenate(
            (Tl0_dc.reshape(M, 1), Tl0_ra.reshape(M, 1)), axis=1)
    elif fit_type in ["T", "t"]:
        jac_mat_ra, jac_mat_dc = Tl0_ra.reshape(M, 1), Tl0_dc.reshape(M, 1)
    elif fit_type in ["S", "s"]:
        jac_mat_ra, jac_mat_dc = -Tl0_dc.reshape(M, 1), Tl0_ra.reshape(M, 1)
    else:
        print("Unknown value for fit_type")
        exit(1)

    for m in range(1, l+1):
        Tlmr_ra, Tlmr_dc, Tlmi_ra, Tlmi_dc = real_vec_sph_harm_proj(
            l, m, T_ra_mat, T_dc_mat)
       # Just to show the relation
       # Slmr_ra, Slmi_ra = -Tlmr_dc, -Tlmr_dc
       # Slmr_dc, Slmr_dc = Tlmr_ra, Tlmr_ra

        # Concatenate the new array and the existing Jacobian matrix
        if fit_type in ["full", "FULL", "Full"]:
            jac_mat_ra = concatenate(
                (jac_mat_ra, Tlmr_ra.reshape(M, 1), -Tlmr_dc.reshape(M, 1),
                 Tlmi_ra.reshape(M, 1), -Tlmi_dc.reshape(M, 1)), axis=1)
            jac_mat_dc = concatenate(
                (jac_mat_dc, Tlmr_dc.reshape(M, 1), Tlmr_ra.reshape(M, 1),
                 Tlmi_dc.reshape(M, 1), Tlmi_ra.reshape(M, 1)), axis=1)
        elif fit_type in ["T", "t"]:
            jac_mat_ra = concatenate(
                (jac_mat_ra, Tlmr_ra.reshape(M, 1), Tlmi_ra.reshape(M, 1)), axis=1)
            jac_mat_dc = concatenate(
                (jac_mat_dc, Tlmr_dc.reshape(M, 1), Tlmi_dc.reshape(M, 1)), axis=1)
        else:
            jac_mat_ra = concatenate(
                (jac_mat_ra, -Tlmr_dc.reshape(M, 1), -Tlmi_dc.reshape(M, 1)), axis=1)
            jac_mat_dc = concatenate(
                (jac_mat_dc, Tlmr_ra.reshape(M, 1), Tlmi_ra.reshape(M, 1)), axis=1)

    # Treat (ra, dc) as two observations(dependent or independent)
    # Combine the Jacobian matrix projection on RA and Decl.
    jac_mat = concatenate((jac_mat_ra, jac_mat_dc), axis=0)

    # Check the shape of the matrix
    if fit_type in ["full", "FULL", "Full"]:
        n1, n2 = 2*M, 4*l+2
    else:
        n1, n2 = 2*M, 2*l+1

    if jac_mat.shape != (n1, n2):
        print("Shape of Jacobian matrix at l={} is ({},{}) "
              "rather than ({},{})".format(
                  l, jac_mat.shape[0], jac_mat.shape[1], n1, n2))
        sys.exit(1)

    return jac_mat


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
        'full' for T- and S-vectors both
        'T' for T-vectors only
        'S' for S-vectors only

    Returns
    ----------
    jac_mat : array of float
        Jacobian matrix  (M, N) (assume N unknows to determine)
    """

    # Calculate all the VSH terms at one time
    T_ra_mat, T_dc_mat = vec_sph_harm_proj(l_max, ra_rad, dc_rad)

    # Usually begins with the first degree
    jac_mat = jac_mat_l_calc(T_ra_mat, T_dc_mat, 1, fit_type)

    for l in range(2, l_max+1):
        new_mat = jac_mat_l_calc(T_ra_mat, T_dc_mat, l, fit_type)
        jac_mat = concatenate((jac_mat, new_mat), axis=1)

    # Check the shape of the Jacobian matrix
    M = len(ra_rad)
    # N = 2 * l_max * (l_max+2)
    if fit_type in ["full", "FULL", "Full"]:
        n1, n2 = 2 * M, 2 * l_max * (l_max+2)
    else:
        n1, n2 = 2 * M, l_max * (l_max+2)

    if jac_mat.shape != (n1, n2):
        print("Shape of Jacobian matrix at l={} is ({},{}) "
              "rather than ({},{})".format(
                  l, jac_mat.shape[0], jac_mat.shape[1], n1, n2))
        sys.exit(1)

    return jac_mat


def nor_mat_calc(dra, ddc, dra_err, ddc_err, ra_rad, dc_rad,
                 ra_dc_cor=None, l_max=1, fit_type="full",
                 save_mat=False, suffix=""):
    """Cacluate the normal and right-hand-side matrix for LSQ analysis.

    Parameters
    ----------
    dra_err/ddc_err : array of float
        formal uncertainty of dRA(*cos(dc_rad))/dDE
    ra_rad/dc_rad : array of float
        Right ascension/Declination in radian
    ra_dc_cor : array of float
        correlation coefficient between dRA and dDE, default is None
    l_max : int
        maximum degree
    fit_type : string
        flag to determine which parameters to be fitted
        'full' for T- and S-vectors both
        'T' for T-vectors only
        'S' for S-vectors only
    suffix : string
        suffix for output matrix file
    save_mat : boolean
        flag to determine if the Jacobian matrix should be saved

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
    wgt_mat = wgt_mat_calc(dra_err, ddc_err, ra_dc_cor)

    # Jac_mat_T * Wgt_mat
    mul_mat = np.dot(np.transpose(jac_mat), wgt_mat)

    # Calculate normal matrix A
    nor_mat = np.dot(mul_mat, jac_mat)

    # Calculate right-hand-side matrix b
    res_mat = concatenate((dra, ddc), axis=0)
    rhs_mat = np.dot(mul_mat, res_mat)

    # Save matrice for later use
    if save_mat:
        np.save('/tmp/jac_mat_{:s}.npy'.format(suffix), jac_mat)
        # np.save('/tmp/wgt_mat_{:s}.npy'.format(suffix), wgt_mat)
        np.save('/tmp/mul_mat_{:s}.npy'.format(suffix), mul_mat)
        np.save('/tmp/nor_mat_{:s}.npy'.format(suffix), nor_mat)

    return nor_mat, rhs_mat


def predict_mat_calc(pmt, ra, dc, l_max, fit_type="full"):
    """Calculate the predicted value

    Parameters
    ----------
    pmt : array
        estimate of unknowns
    ra/dc: array of float
        Right ascension/Declination in radian
    l_max : int
        maximum degree

    Returns
    -------
    dra_pre : array
        predicted offset in RA
    ddc_pre : array
        predicted offset in Declination
    """

    jac_mat = jac_mat_calc(ra, dc, l_max, fit_type)
    dra_ddc = np.dot(jac_mat, pmt)
    num_sou = int(len(dra_ddc)/2)
    dra_pre, ddc_pre = dra_ddc[:num_sou], dra_ddc[num_sou:]

    return dra_pre, ddc_pre


def predict_mat_calc_4_huge(pmt, ra_rad, dc_rad, l_max, fit_type="full", num_iter=None):
    """Calculate the predicted value

    Parameters
    ----------
    pmt : array
        estimate of unknowns
    ra/dc: array of float
        Right ascension/Declination in radian
    l_max : int
        maximum degree

    Returns
    -------
    dra_pre : array
        predicted offset in RA
    ddc_pre : array
        predicted offset in Declination
    """

    if num_iter is None:
        num_iter = 100

    # Divide huge table into small pieces
    div = ra_rad.size // num_iter
    rem = ra_rad.size % num_iter

    # Empty array for use
    dra_pre = np.array([], dtype=float)
    ddc_pre = np.array([], dtype=float)

    if rem:
        drai, ddci = predict_mat_calc(
            pmt, ra_rad[: rem], dc_rad[: rem], l_max, fit_type=fit_type)
        dra_pre = concatenate((dra_pre, drai), axis=0)
        ddc_pre = concatenate((ddc_pre, ddci), axis=0)

    for i in range(div):
        sta = rem + i * num_iter
        end = sta + num_iter

        drai, ddci = predict_mat_calc(
            pmt, ra_rad[sta: end], dc_rad[sta: end], l_max, fit_type=fit_type)
        dra_pre = concatenate((dra_pre, drai), axis=0)
        ddc_pre = concatenate((ddc_pre, ddci), axis=0)

    return dra_pre, ddc_pre


def residual_calc(dra, ddc, ra_rad, dc_rad, pmt, l_max, fit_type="full",
                  num_iter=None):
    """Calculate post-fit residual

    Parameters
    ----------
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences
    ra_rad/dc_rad : array of float
        Right ascension/Declination in radian
    pmt : array of float
        estimation of (d1, d2, d3, r1, r2, r3)
    fit_type : string
        flag to determine which parameters to be fitted
        'full' for T- and S-vectors both
        'T' for T-vectors only
        'S' for S-vectors only
    l_max : int
        maximum degree

    Returns
    -------
    dra_r : array
        residual in RA
    ddc_r : array
        residual in Declination
    """

    dra_pre, ddc_pre = predict_mat_calc_4_huge(
        pmt, ra_rad, dc_rad, l_max, fit_type, num_iter)

    dra_r = dra - dra_pre
    ddc_r = ddc - ddc_pre

    return dra_r, ddc_r


def nor_eq_sol(dra, ddc, dra_err, ddc_err, ra_rad, dc_rad, ra_dc_cor=None,
               l_max=1, fit_type="full", num_iter=None, calc_res=True):
    """The 1st degree of VSH function: glide and rotation.

    Parameters
    ----------
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences
    dra_err/ddc_err : array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc
    ra_rad/dc_rad : array of float
        Right ascension/Declination in radian
    ra_dc_cov/ra_dc_cor : array of float
        covariance/correlation coefficient between dra and ddc^2, default is None
    fit_type : string
        flag to determine which parameters to be fitted
        'full' for T- and S-vectors both
        'T' for T-vectors only
        'S' for S-vectors only

    Returns
    ----------
    pmt : array of float
        estimation of (d1, d2, d3, r1, r2, r3)
    sig : array of float
        uncertainty of x
    cor_mat : matrix
        matrix of correlation coefficient.
    """

    # Maxium number of sources processed per time
    # According to my test, 100 should be a good choice
    if num_iter is None:
        num_iter = 100

    div = dra.size // num_iter
    rem = dra.size % num_iter

    if rem:
        if not ra_dc_cor is None:
            A, b = nor_mat_calc(dra[: rem], ddc[: rem], dra_err[: rem], ddc_err[: rem],
                                ra_rad[: rem], dc_rad[: rem], ra_dc_cor=ra_dc_cor[: rem],
                                l_max=l_max, fit_type=fit_type)
        else:
            A, b = nor_mat_calc(dra[: rem], ddc[: rem], dra_err[: rem], ddc_err[: rem],
                                ra_rad[: rem], dc_rad[: rem], l_max=l_max, fit_type=fit_type)

    else:
        A, b = 0, 0

    for i in range(div):
        sta = rem + i * num_iter
        end = sta + num_iter

        if not ra_dc_cor is None:
            An, bn = nor_mat_calc(dra[sta: end], ddc[sta: end], dra_err[sta: end],
                                  ddc_err[sta: end], ra_rad[sta: end], dc_rad[sta: end],
                                  ra_dc_cor=ra_dc_cor[sta: end], l_max=l_max, fit_type=fit_type)
        else:
            An, bn = nor_mat_calc(dra[sta: end], ddc[sta: end], dra_err[sta: end],
                                  ddc_err[sta: end], ra_rad[sta: end], dc_rad[sta: end],
                                  l_max=l_max, fit_type=fit_type)
        A = A + An
        b = b + bn

    # Solve the equations.
    pmt = np.linalg.solve(A, b)

    # Covariance.
    cov_mat = np.linalg.inv(A)

    # Formal uncertainty and correlation coefficient
    sig, cor_mat = cov_to_cor(cov_mat)

    # Calculate residuals
    if calc_res:
        dra_r, ddc_r = residual_calc(dra, ddc, ra_rad, dc_rad, pmt, l_max,
                                     fit_type=fit_type, num_iter=num_iter)

        # Return the result.
        return pmt, sig, cor_mat, dra_r, ddc_r
    else:
        return pmt, sig, cor_mat


def nor_mat_calc_for_cache(dra_err, ddc_err, ra_rad, dc_rad,
                           ra_dc_cor=None, l_max=1, fit_type="full", suffix=""):
    """Cacluate the normal and right-hand-side matrix for LSQ analysis.

    Parameters
    ----------
    dra_err/ddc_err : array of float
        formal uncertainty of dRA(*cos(dc_rad))/dDE
    ra_rad/dc_rad : array of float
        Right ascension/Declination in radian
    ra_dc_cor : array of float
        correlation coefficient between dRA and dDE, default is None
    l_max : int
        maximum degree
    fit_type : string
        flag to determine which parameters to be fitted
        'full' for T- and S-vectors both
        'T' for T-vectors only
        'S' for S-vectors only
    suffix : string
        suffix for output matrix file
    save_mat : boolean
        flag to determine if the Jacobian matrix should be saved

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
    wgt_mat = wgt_mat_calc(dra_err, ddc_err, ra_dc_cor)

    # Jac_mat_T * Wgt_mat
    mul_mat = np.dot(np.transpose(jac_mat), wgt_mat)

    # Calculate normal matrix A
    nor_mat = np.dot(mul_mat, jac_mat)

    # Save matrice for later use
    np.save("/tmp/jac_mat_{:s}.npy".format(suffix), jac_mat)
    # np.save("/tmp/wgt_mat_{:s}.npy".format(suffix), wgt_mat)
    np.save("/tmp/mul_mat_{:s}.npy".format(suffix), mul_mat)
    np.save("/tmp/nor_mat_{:s}.npy".format(suffix), nor_mat)


def cache_mat_calc(dra, ddc, dra_err, ddc_err, ra_rad, dc_rad, ra_dc_cor=None,
                   l_max=1, fit_type="full", num_iter=None):
    """Calculate cache matrix for future use

    Parameters
    ----------
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences
    dra_err/ddc_err : array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc
    ra_rad/dc_rad : array of float
        Right ascension/Declination in radian
    ra_dc_cov/ra_dc_cor : array of float
        covariance/correlation coefficient between dra and ddc, default is None
    fit_type : string
        flag to determine which parameters to be fitted
        'full' for T- and S-vectors both
        'T' for T-vectors only
        'S' for S-vectors only

    Returns
    ----------
    pmt : array of float
        estimation of (d1, d2, d3, r1, r2, r3)
    sig : array of float
        uncertainty of x
    cor_mat : matrix
        matrix of correlation coefficient.
    """

    # Maxium number of sources processed per time
    # According to my test, 100 should be a good choice
    if num_iter is None:
        num_iter = 100

    div = dra.size // num_iter
    rem = dra.size % num_iter
    suffix_array = []

    if rem:
        suffix_array.append("{:05d}".format(0))
        if not ra_dc_cor is None:
            nor_mat_calc_for_cache(dra_err[: rem], ddc_err[: rem],
                                   ra_rad[: rem], dc_rad[: rem],
                                   ra_dc_cor=ra_dc_cor[: rem],
                                   l_max=l_max, fit_type=fit_type,
                                   suffix=suffix_array[0])
        else:
            nor_mat_calc_for_cache(dra_err[: rem], ddc_err[: rem],
                                   ra_rad[: rem], dc_rad[: rem],
                                   l_max=l_max, fit_type=fit_type,
                                   suffix=suffix_array[0])

    for i in range(div):
        sta = rem + i * num_iter
        end = sta + num_iter
        suffix_array.append("{:05d}".format(i+1))

        if not ra_dc_cor is None:
            nor_mat_calc_for_cache(dra_err[sta: end], ddc_err[sta: end],
                                   ra_rad[sta: end], dc_rad[sta: end],
                                   ra_dc_cor=ra_dc_cor[sta: end],
                                   l_max=l_max, fit_type=fit_type,
                                   suffix=suffix_array[-1])
        else:
            nor_mat_calc_for_cache(dra_err[sta: end], ddc_err[sta: end],
                                   ra_rad[sta: end], dc_rad[sta: end],
                                   l_max=l_max, fit_type=fit_type,
                                   suffix=suffix_array[-1])

    return suffix_array


def nor_mat_calc_from_cahce(dra, ddc, suffix):
    """Cacluate t he normal and right-hand-side matrix for LSQ analysis.

    Parameters
    ----------
    dra/ddc : array of float
        dRA(*cos(dc_rad))/dDE
    suffix : string
        suffix for output matrix file

    Returns
    ----------
    nor_mat : array of float
        normal matrix
    rhs_mat : array of float
        right-hand-side matrix
    """

    # Jac_mat_T * Wgt_mat
    mul_mat = np.load("/tmp/mul_mat_{:s}.npy".format(suffix))

    # Calculate normal matrix A
    nor_mat = np.load("/tmp/nor_mat_{:s}.npy".format(suffix))

    # Calculate right-hand-side matrix b
    res_mat = concatenate((dra, ddc), axis=0)
    rhs_mat = np.dot(mul_mat, res_mat)

    return nor_mat, rhs_mat


def predict_mat_calc_from_cache(pmt, suffix):
    """Calculate the predicted value from cached Jacobian matrix

    Parameters
    ----------
    pmt : array
        estimate of unknowns
    suffix : string
        suffix for finding corresponding Jacobian matrix

    Returns
    -------
    dra_pre : array
        predicted offset in RA
    ddc_pre : array
        predicted offset in Declination
    """

    jac_mat = np.load("/tmp/jac_mat_{:s}.npy".format(suffix))
    dra_ddc = np.dot(jac_mat, pmt)
    num_sou = int(len(dra_ddc)/2)
    dra_pre, ddc_pre = dra_ddc[:num_sou], dra_ddc[num_sou:]

    return dra_pre, ddc_pre


def residual_calc_from_cache(dra, ddc, pmt, suffix_array):
    """Calculate post-fit residual

    Parameters
    ----------
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences
    pmt : array of float
        estimation of (d1, d2, d3, r1, r2, r3)
    suffix_array : array fo string
        suffix to find the corresponding cache

    Returns
    -------
    dra_r : array
        residual in RA
    ddc_r : array
        residual in Declination
    """

    # Calculate predict value
    dra_pre = np.array([], dtype=float)
    ddc_pre = np.array([], dtype=float)

    for suffix in suffix_array:
        dra_prei, ddc_prei = predict_mat_calc_from_cache(pmt, suffix)
        dra_pre = concatenate((dra_pre, dra_prei))
        ddc_pre = concatenate((ddc_pre, ddc_prei))

    dra_r, ddc_r = dra - dra_pre, ddc - ddc_pre

    return dra_r, ddc_r


def nor_eq_sol_from_cache(dra, ddc, suffix_array, num_iter=None, calc_res=True):
    """Calculate cache matrix for future use

    Parameters
    ----------
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences

    Returns
    ----------
    pmt : array of float
        estimation of (d1, d2, d3, r1, r2, r3)
    sig : array of float
        uncertainty of x
    cor_mat : matrix
        matrix of correlation coefficient.
    """

    # Maxium number of sources processed per time
    # According to my test, 100 should be a good choice
    if num_iter is None:
        num_iter = 100

    div = dra.size // num_iter
    rem = dra.size % num_iter
    suffix_array = []
    index = 0

    if rem:
        suffix = suffix_array[index]
        index += 1
        A, b = nor_mat_calc_from_cahce(dra[:rem], ddc[:rem], suffix)
    else:
        A, b = 0, 0

    for i in range(div):
        sta = rem + i * num_iter
        end = sta + num_iter
        suffix = suffix_array[index]
        index += 1

        An, bn = nor_mat_calc_from_cache(dra[sta: end], ddc[sta: end], suffix)

    # Solve the equations.
    pmt = np.linalg.solve(A, b)

    # Covariance.
    cov_mat = np.linalg.inv(A)
    # Formal uncertainty and correlation coefficient
    sig, cor_mat = cov_to_cor(cov_mat)

    # Calculate residuals
    if calc_res:
        dra_r, ddc_r = residual_calc_from_cache(dra, ddc, pmt, suffix_array)

        # Return the result.
        return pmt, sig, cor_mat, dra_r, ddc_r
    else:
        return pmt, sig, cor_mat


def rm_cache_mat(suffix_array):
    """Remove cache

    """

    for suffix in suffix_array:
        # Delete Jacobian matrix file
        os.system("rm /tmp/jac_mat_{:s}.npy".format(suffix))

        # Delete multiplication matrix file
        os.system("rm /tmp/mul_mat_{:s}.npy".format(suffix))

        # Delete normal matrix file
        os.system("rm /tmp/nor_mat_{:s}.npy".format(suffix))
# --------------------------------- END --------------------------------
