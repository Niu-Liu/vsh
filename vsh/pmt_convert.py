#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: pmt_convert.py
"""
Created on Wed Nov 18 10:28:30 2020

@author: Neo(niu.liu@nju.edu.cn)

Convert S&T parameters into familiar rotation/glide/quadrupolar ones
"""

import functools
import sys
import numpy as np
# My progs
from .matrix_calc import cor_to_cov, cov_to_cor

# Scaling factors between S/T and rotation/glide/quadrupolar parameters
FAC1 = np.sqrt(3. / 4 / np.pi)
FAC2 = np.sqrt(3. / 8 / np.pi)
FAC3 = np.sqrt(15. / 32 / np.pi)
FAC4 = np.sqrt(5. / 4 / np.pi)
FAC5 = np.sqrt(5. / 16 / np.pi)


# -----------------------------  MAIN -----------------------------
def st_to_rotgld(pmt, err, cor_mat, fit_type="full"):
    """Transformation from degree 1 S&T-vector to rotation&glide vector

    pmt = (t10, s10, t11r, s11r, t11i, s11i)
    gr_vec = (G1, G2, G3, R1, R2, R3)

    This transformation is compatible with these codes written in VSHdeg01_cor.

    Parameters
    ----------
    pmt : array of (6,)
        estimation of T&S coefficient
    err : array of (6,)
        formal uncertainty of pmt
    cor_mat : array of (6, 6)
        correlation coefficient matrix

    Returns
    -------
    gr_vec : array of (6,)
        estimation of glide and rotation parameters
    gr_err : array of (6,)
        formal uncertainty of glide and rotation
    gr_cor_mat : array of (6, 6)
        correlation coefficient matrix
    """

    if fit_type in ["full", "FULL", "Full"]:
        len_vec = 6
    elif fit_type in ["S", "T", "s", "t"]:
        len_vec = 3

    if len(pmt) != len_vec or len(err) != len_vec:
        print("Length of S&T-vector should be equal to {}".format(len_vec))
        sys.exit()

    # Transformation matrix
    # Assume from (T10, T11r, T11i) to (R1, R2, R3)
    #    R1 = -FAC1 * T11r
    #    R2 = FAC1 * T11i
    #    R3 = FAC2 * T10
    # It should be same from (S10, S11r, S11i) to (G1, G2, G3)

    if fit_type in ["full", "FULL", "Full"]:
        TSF_MAT = np.array([[0, 0, 0, -FAC1, 0, 0],
                            [0, 0, 0, 0, 0, FAC1],
                            [0, FAC2, 0, 0, 0, 0],
                            [0, 0, -FAC1, 0, 0, 0],
                            [0, 0, 0, 0, FAC1, 0],
                            [FAC2, 0, 0, 0, 0, 0]])

    elif fit_type in ["S", "s", "T", "t"]:
        TSF_MAT = np.array([[0, -FAC1, 0],
                            [0, 0, FAC1],
                            [FAC2, 0, 0]])

    # Glide &Rotation vector
    gr_vec = np.dot(TSF_MAT, pmt)

    # Formal uncertainty and correlation coefficient
    cov_mat = cor_to_cov(err, cor_mat)
    gr_cov_mat = functools.reduce(np.dot, [TSF_MAT, cov_mat, TSF_MAT.T])
    gr_err, gr_cor_mat = cov_to_cor(gr_cov_mat)

    return gr_vec, gr_err, gr_cor_mat


def st_to_rotgldquad(pmt, err, cor_mat, fit_type="full"):
    """Transformation from degree 2 S/T-vectors to rotation/glide/quadrupolar vector

    pmt = (t10, s10, t11r, s11r, t11i, s11i,
           t20, s20,
           t21r, s21r, t21i, s21i,
           t22r, s22r, t22i, s22i)

    grq_vec = (G1, G2, G3, R1, R2, R3,
               E22R, E22I, E21R, E21I, E20,
               M22R, M22I, M21R, M21I, M20)
    """

    if fit_type in ["full", "FULL", "Full"]:
        len_vec = 16
    elif fit_type in ["S", "T", "s", "t"]:
        len_vec = 8

    if len(pmt) != len_vec or len(err) != len_vec:
        print("Length of S&T-vector should be equal to 16")
        sys.exit()

    # Transformation matrix
    # For the first degree, refer to line 59-53
    # Assume from (T20, T21r, T21i, T22r, T22i) to (M20, M21R, M21I, M22R,
    # M22I)
    #    M20  = FAC3 * T20
    #    M21R = FAC4 * T21r
    #    M21I = FAC4 * T21i
    #    M22R = FAC5 * T22r
    #    M22I = FAC5 * T22i
    # It should be same from (S20, S21r, S21i, S22r, S22i) to (E20, E21R, E21I, E22R,
    # E22I)

    if fit_type in ['full', 'FULL', 'Full']:
        TSF_MAT = np.array([[0, 0, 0, -FAC1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, FAC1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, FAC2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, -FAC1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, FAC1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [FAC2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FAC5, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FAC5],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, FAC4, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FAC4, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, FAC3, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FAC5, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FAC5, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, FAC4, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FAC4, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, FAC3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    else:
        TSF_MAT = np.array([[0, -FAC1, 0, 0, 0, 0, 0, 0],
                            [0, 0, FAC1, 0, 0, 0, 0, 0],
                            [FAC2, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, FAC5, 0, 0],
                            [0, 0, 0, 0, 0, 0, FAC5, 0],
                            [0, 0, 0, FAC4, 0, 0, 0, 0],
                            [0, 0, 0, 0, FAC4, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, FAC3]])

    # Glide &Rotation vector
    grq_vec = np.dot(TSF_MAT, pmt)

    # Formal uncertainty and correlation coefficient
    cov_mat = cor_to_cov(err, cor_mat)
    grq_cov_mat = functools.reduce(np.dot, [TSF_MAT, cov_mat, TSF_MAT.T])
    grq_err, grq_cor_mat = cov_to_cor(grq_cov_mat)

    return grq_vec, grq_err, grq_cor_mat


def calc_vec_apex(pmt, err, cor_mat):
    """Calculate the apex and amplitude for a given 3-D vector.

    Parameters
    ----------
    pmt : array of (3,)
        estimation of 3-D vector
    err : array of (3,)
        formal uncertainty of pmt
    cor_mat : array of (3, 3)
        correlation coefficient matrix

    Returns
    ----------
    g/g_err : amplitude and its uncertainty
    RA_deg/RA_err : right ascension and its uncertainty in degree
    DC_deg/DC_err : declination and its uncertainty in degree
    """

    # Calculate the parameters.
    # Amplitude
    g = np.sqrt(np.sum(pmt ** 2))

    # Right ascension
    RA_rad = np.arctan2(pmt[1], pmt[0])
    # arctan2 function result is always between -pi and pi.
    # However, RA should be between 0 and 2*pi.
    if RA_rad < 0:
        RA_rad += 2 * np.pi

    # Declination
    DC_rad = np.arctan2(pmt[2], np.sqrt(pmt[0]**2 + pmt[1]**2))

    # radian -> degree
    RA_deg = np.rad2deg(RA_rad)
    DC_deg = np.rad2deg(DC_rad)

    # Covariance matrix
    cov_mat = cor_to_cov(err, cor_mat)

    # Jacobian matrix of par(g, ra, dec) / par(g1, g2, g3)
    par_g = pmt / g
    gxy2 = pmt[0]**2 + pmt[1]**2
    gxy = np.sqrt(gxy2)
    par_RA = np.array([-pmt[1], pmt[0], 0]) / gxy2
    par_DC = np.array([-pmt[0] * pmt[2],
                       -pmt[1] * pmt[2],
                       gxy2]) / g**2 / gxy

    jac_mat = np.vstack((par_g, par_RA, par_DC))

    # New covaraince matrix
    new_cov_mat = functools.reduce(np.dot, [jac_mat, cov_mat, jac_mat.T])

    # Formal uncertainty and correlation
    new_err, new_cor_mat = cov_to_cor(new_cov_mat)
    g_err = new_err[0]
    # radian -> degree
    RA_err = np.rad2deg(new_err[1])
    DC_err = np.rad2deg(new_err[2])

    new_pmt = [g, RA_deg, DC_deg]
    new_err = [g_err, RA_err, DC_err]

    return new_pmt, new_err


def convert_ts_to_rgq(pmt, err, cor_mat, l_max, fit_type="full"):
    """Print Rotation/glide converted from t_lm and s_lm

    Parameters
    ----------
    pmt : array-like of float
        t_lm/s_lm coefficient
    err : array-like of float
        formal uncertaity of pmt
    cor_mat : matrix-like, float
        correlation coefficient matrix
    l_max : int
        maximum degree
    fit_type : string
        flag to determine which parameters to be fitted
        "full" for T- and S-vectors both
        "T" for T-vectors only
        "S" for S-vectors only

    Returns
    -------
    pmt1 : array of float
        estimation of glide/rotation/quadrupolar terms in dex
    sig1 : array of float
        formal uncertainty of pmt1 in dex
    cor_mat1 : matrix
        matrix of correlation coefficient among pmt1.
    """

    output = {}

    if l_max == 1:
        pmt1, err1, cor_mat1 = st_to_rotgld(pmt[:6], err[:6], cor_mat[:6], fit_type)
    elif l_max >= 2:
        pmt1, err1, cor_mat1 = st_to_rotgldquad(
            pmt[:16], err[:16], cor_mat[:16, :16], fit_type)
    else:
        print(" l_max = {} is an invalid input.".format(l_max))
        sys.exit(1)

    if fit_type in ["T", "t"]:
        pmt1_out = np.hstack(([0, 0, 0], pmt1))
        err1_out = np.hstack(([0, 0, 0], err1))
        r_vec, r_vec_err = calc_vec_apex(pmt1, err1, cor_mat1)
        g_vec, g_vec_err = np.zeros(3), np.zeros(3)

    elif fit_type in ["S", "s"]:
        pmt1_out = np.hstack((pmt1, [0, 0, 0]))
        err1_out = np.hstack((err1, [0, 0, 0]))
        r_vec, r_vec_err = np.zeros(3), np.zeros(3)
        g_vec, g_vec_err = calc_vec_apex(pmt1, err1, cor_mat1)

    elif fit_type in ["full", "Full"]:
        pmt1_out = pmt1
        err1_out = err1

        r_vec, r_vec_err = calc_vec_apex(pmt1[3:6], err1[3:6], cor_mat1[3:6, 3:6])
        g_vec, g_vec_err = calc_vec_apex(pmt1[:3], err1[:3], cor_mat1[:3, :3])
    else:
        sys.exit(1)

    print("")
    print("Convert t_lm/s_lm at l=1 into rotation/glide vector")
    print("--------------------------------------------------------------------")
    print("           Glide [dex]      "
          "           Rotation [dex]   ")
    print("  G1         G2        G3       "
          "  R1         R2        R3       ")
    print("--------------------------------------------------------------------")
    print("{0:+4.0f} {6:4.0f}  {1:+4.0f} {7:4.0f} "
          "{2:+4.0f} {8:4.0f}  {3:+4.0f} {9:4.0f} "
          "{4:+4.0f} {10:4.0f}  {5:+4.0f} {11:4.0f}  ".format(
              *pmt1_out[:6], *err1_out[:6]))
    print("--------------------------------------------------------------------")

    output["pmt1"] = pmt1
    output["sig1"] = err1
    output["cor_mat1"] = cor_mat1

    output["R"], output["R_err"] = r_vec[0], r_vec_err[0]
    output["R_ra"], output["R_ra_err"] = r_vec[1], r_vec_err[1]
    output["R_dec"], output["R_dec_err"] = r_vec[2], r_vec_err[2]

    output["G"], output["G_err"] = g_vec[0], g_vec_err[0]
    output["G_ra"], output["G_ra_err"] = g_vec[1], g_vec_err[1]
    output["G_dec"], output["G_dec_err"] = g_vec[2], g_vec_err[2]

    return output


def add_rgq_to_output(output, output_ext, l_max):
    """Add rotation/glide/quadrupolar terms to output

    Parameter
    ---------
    output : dict-like, float
        -pmt : array of float
            estimation of (d1, d2, d3, r1, r2, r3) in dex
        -sig : array of float
            uncertainty of x in dex
        -cor_mat : matrix
            matrix of correlation coefficient.
    pmt1 : array of float
        estimation of glide/rotation/quadrupolar terms in dex
    sig1 : array of float
        formal uncertainty of pmt1 in dex
    cor_mat1 : matrix
        matrix of correlation coefficient among pmt1.
    l_max : int
        maximum degree

    Return
    ------
    output : dict-like, float
        -pmt : array of float
            estimation of (d1, d2, d3, r1, r2, r3) in dex
        -sig : array of float
            uncertainty of x in dex
        -cor_mat : matrix
            matrix of correlation coefficient.
        -pmt1 : array of float
            estimation of glide/rotation/quadrupolar terms in dex
        -sig1 : array of float
            formal uncertainty of pmt1 in dex
        -cor_mat1 : matrix
            matrix of correlation coefficient among pmt1.
    """

    if l_max == 1:
        output["pmt1"] = output_ext["pmt1"]
        output["sig1"] = output_ext["sig1"]
        output["cor1"] = output_ext["cor_mat1"]
        output["note"] = output["note"] + [
            "pmt1: glide+rotation\n"
            "sig1: formal error of glide/rotation\n"
            "cor1: correlation coeficient matrix of glide/rotation\n"][0]

    elif l_max >= 2:
        output["pmt2"] = output_ext["pmt1"]
        output["sig2"] = output_ext["sig1"]
        output["cor2"] = output_ext["sig1"]
        output["note"] = output["note"] + [
            "pmt2: glide+rotation+quadrupolar\n"
            "sig2: formal error of glide/rotation/quadrupolar\n"
            "cor2: correlation coeficient matrix of glide/rotation/quad\n"][0]
    else:
        print(" l_max = {} is an invalid input.".format(l_max))
        sys.exit(1)

    output["R"], output["R_err"] = output_ext["R"], output_ext["R_err"]
    output["R_ra"], output["R_ra_err"] = output_ext["R_ra"], output_ext["R_ra_err"]
    output["R_dec"], output["R_dec_err"] = output_ext["R_dec"], output_ext["R_dec_err"]

    output["G"], output["G_err"] = output_ext["G"], output_ext["G_err"]
    output["G_ra"], output["G_ra_err"] = output_ext["G_ra"], output_ext["G_ra_err"]
    output["G_dec"], output["G_dec_err"] = output_ext["G_dec"], output_ext["G_dec_err"]

    output["note"] = output["note"] + [
        "R/G: amplitude of rotation/Glide\n"
        "R_ra, R_dec: apex of rotation vector\n"
        "G_ra, G_dec: apex of glide vector\n"][0]

    return output


def main():
    """Not implemented yet
    """
    pass


if __name__ == "__main__":
    main()
# --------------------------------- END --------------------------------
