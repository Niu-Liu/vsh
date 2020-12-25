#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: vec_sph_harm.py
"""
Created on Thu May  7 20:22:39 2020

@author: Neo(liuniu@smail.nju.edu.cn)

Calculate the vector spherical harmonics (VSH).

For an introduction of the VSH, please refer to F. Mignard and S. Klioner, A&A 547, A59 (2012).
DOI: 10.1051/0004-6361/201219927.

24/11/2020: use scipy.special.lpmn to generate the associated Legendre functions
            and their derivatives
"""

import sys
import numpy as np
from numpy import sqrt, pi, sin, cos, exp
from math import factorial
from scipy.special import lpmn


# -----------------------------  FUNCTIONS -----------------------------
def mat_2_seq(mat):
    """Collect all elements above the diagonal of a matrix
    """

    mat_len = mat.shape[0]
    # skip (0,0)
    seq = mat[1, :2]

    # Loop to collect other elements
    for i in range(2, mat_len):
        seq = np.concatenate((seq, mat[i, :i+1]))

    return seq


def vec_sph_harm_proj(l_max, ra, dc, sph_type="T"):
    """Calculate the vsh function at (ra, dec)

    Note that ra and dec should be scalar rather than vector

    Parameters
    ----------
    l_max: int
        maximum degree of the harmonics
    ra, dc: float
        equatorial coordinates in the unit of radian
    sph_type: string
        "T" or "S", might be used in the future

    Returns
    -------
    T_ra, T_dc: complex
        Projection of T_lm vector on the e_ra and e_dec vectors.
#   S_ra, S_dc: complex
#       Projection of S_lm vector on the e_ra and e_dec vectors.
    """

    # In some cases that l_max is not an integer
    l_max = int(l_max)

    # Calculate A_mn(x) and B_mn(x)
    x = sin(dc)
    fac = np.sqrt(1-x*x)

    # Initialize the Amn and Bmn
    A_mat = np.zeros((l_max+1, l_max+1))
    B_mat = np.zeros((l_max+1, l_max+1))
    B_mat[1, 1] = 1
    B_mat[1, 0] = 0

    # Generate the sequence of Bmn
    for l in range(2, l_max+1):
        for m in range(l+1)[::-1]:
            if m:
                if l == m:
                    B_mat[l, m] = fac * (2*m-1) * m / (m-1) * B_mat[m-1, m-1]
                elif l == m+1:
                    B_mat[l, m] = (2*m+1) * x * B_mat[m, m]
                else:
                    B_mat[l, m] = ((2*l-1)*x*B_mat[l-1, m] -
                                   (l-1+m)*B_mat[l-2, m]) / (l-m)
            else:
                B_mat[l, m] = 0

    # Calculate Amn
    for l in range(1, l_max+1):
        for m in range(l+1):
            if m:
                A_mat[l, m] = (-x*l*B_mat[l, m]+(l+m)*B_mat[l-1, m]) / m
            else:
                A_mat[l, m] = fac * B_mat[l, 1]

    # Calculate the coefficient in Eqs. (B.9-B.10)
    fac_mat = np.zeros((l_max+1, l_max+1)) * (0+0j)
    for l in range(1, l_max+1):
        for m in range(l+1):
            fac = (2*l+1) / (l*l+l) / (4*pi) * factorial(l-m) / factorial(l+m)
            fac = (-1)**m * sqrt(fac) * exp(complex(0, m*ra))
            fac_mat[l, m] = fac

    # Projection on ra and decl. direction
    T_ra_mat = fac_mat * A_mat
    T_dc_mat = fac_mat * B_mat * (0-1j)

    # Matrix -> Array
    T_ra = mat_2_seq(T_ra_mat)
    T_dc = mat_2_seq(T_dc_mat)

    # Actually S-vector could be derived from T-vector
#    S_ra = facB * (0+1j)
#    S_dc = facA
    if sph_type == "T":
        return T_ra, T_dc
    elif sph_type == "S":
        S_ra_mat = -T_dc_mat
        S_dc_mat = T_ra_mat
        return S_ra_mat, S_dc_mat
    else:
        print("The sph_type can only be 'T' or 'S', Please check it")
        sys.exit()


def real_vec_sph_harm_proj(l_max, ra, dc):
    """Calculate the real (not complex) vsh function of (l,m) at x used.

    VSH functions used for real expansion according to Eq.(30) in the reference.

    Please note that the imaginary part has the opposite sign to that of vec_sph_harm.
    And never forget the factor of 2.

    Parameters
    ----------
    l_max: int
        maximum degree of the harmonics
    ra, dc: float
        equatorial coordinates in the unit of radian

    Returns
    -------
    T_ra_r/i, T_dc_r/i: real vector
        Projection of T_lm (l=1:l_max,m=0:l) vector on the e_ra and e_dec vectors.
    """

    T_ra, T_dc = vec_sph_harm_proj(l_max, ra, dc)

    # Check the shape of T_ra
    T_len = len(T_ra)
    exp_len = (l_max + 3) * l_max / 2
    if T_len != exp_len:
        print("Shape of vector harmonics sequnence at l_max={} is {}"
              "rather than {}".format(l_max, T_len, exp_len))
        sys.exit()

    # According to Eq.(30)
    T_ra_r, T_ra_i = 2*np.real(T_ra), -2*np.imag(T_ra)
    T_dc_r, T_dc_i = 2*np.real(T_dc), -2*np.imag(T_dc)

    return T_ra_r, T_dc_r, T_ra_i, T_dc_i
# --------------------------------- END --------------------------------
