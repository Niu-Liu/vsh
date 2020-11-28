#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: pmt_convert.py
"""
Created on Wed Nov 18 10:28:30 2020

@author: Neo(niu.liu@nju.edu.cn)

Convert S&T parameters into familiar rotation/glide/quadrupolar ones
"""

import sys
import numpy as np
# My progs
from matrix_calc import cor_to_cov, cov_to_cor


# Scaling factors between S/T and rotation/glide/quadrupolar parameters
FAC1 = np.sqrt(3. / 4 / np.pi)
FAC2 = np.sqrt(3. / 8 / np.pi)
FAC3 = np.sqrt(15. / 32 / np.pi)
FAC4 = np.sqrt(5. / 4 / np.pi)
FAC5 = np.sqrt(5. / 16 / np.pi)


# -----------------------------  MAIN -----------------------------
def st_to_rotgld(pmt, err, cor_mat):
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

    if len(pmt) != 6 or len(err) != 6:
        print("Length of S&T-vector should be equal to 6")
        sys.exit()

    # Transformation matrix
    # Assume from (T10, T11r, T11i) to (R1, R2, R3)
    #    R1 = -FAC1 * T11r
    #    R2 = FAC1 * T11i
    #    R3 = FAC2 * T10
    # It should be same from (S10, S11r, S11i) to (G1, G2, G3)

    TSF_MAT = np.array([[0, 0, 0, -FAC1, 0, 0],
                        [0, 0, 0, 0, 0, FAC1],
                        [0, FAC2, 0, 0, 0, 0],
                        [0, 0, -FAC1, 0, 0, 0],
                        [0, 0, 0, 0, FAC1, 0],
                        [FAC2, 0, 0, 0, 0, 0]])

    # Glide &Rotation vector
    gr_vec = np.dot(TSF_MAT, pmt)

    # Formal uncertainty and correlation coefficient
    cov_mat = cor_to_cov(err, cor_mat)
    TSF_MAT_T = np.transpose(TSF_MAT)
    gr_cov_mat = np.dot(np.dot(TSF_MAT, cov_mat), TSF_MAT_T)
    gr_err, gr_cor_mat = cov_to_cor(gr_cov_mat)

    return gr_vec, gr_err, gr_cor_mat


def st_to_rotgldquad(pmt, err, cor_mat):
    """Transformation from degree 2 S/T-vectors to rotation/glide/quadrupolar vector

    pmt = (t10, s10, t11r, s11r, t11i, s11i,
           t20, s20,
           t21r, s21r, t21i, s21i,
           t22r, s22r, t22i, s22i)

    grq_vec = (G1, G2, G3, R1, R2, R3,
               E22R, E22I, E21R, E21I, E20,
               M22R, M22I, M21R, M21I, M20)

    """

    if len(pmt) != 16 or len(err) != 16:
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

    # Glide &Rotation vector
    grq_vec = np.dot(TSF_MAT, pmt)

    # Formal uncertainty and correlation coefficient
    cov_mat = cor_to_cov(err, cor_mat)
    TSF_MAT_T = np.transpose(TSF_MAT)
    grq_cov_mat = np.dot(np.dot(TSF_MAT, cov_mat), TSF_MAT_T)
    grq_err, grq_cor_mat = cov_to_cor(grq_cov_mat)

    return grq_vec, grq_err, grq_cor_mat


def convert_ts_to_rotgli(pmt, err, cor_mat):
    """Print Rotation/glide converted from t_lm and s_lm
    """

    pmt1, err1, cor_mat1 = st_to_rotgld(pmt[:6], err[:6], cor_mat[:6])

    print("")
    print("Convert t_lm/s_lm at l=1 into rotation/glide vector")
    print("--------------------------------------------------------------------")
    print("           Glide [uas]             "
          "           Rotation [uas]          ")
    print("  G1         G2        G3       "
          "  R1         R2        R3        ")
    print("--------------------------------------------------------------------")
    print("{0:4.0f} {6:4.0f}  {1:4.0f} {7:4.0f} "
          "{2:4.0f} {8:4.0f}  {3:4.0f} {9:4.0f} "
          "{4:4.0f} {10:4.0f}  {5:4.0f} {11:4.0f}  ".format(*pmt1[:6], *err1[:6]))
    print("--------------------------------------------------------------------")

# --------------------------------- END --------------------------------
