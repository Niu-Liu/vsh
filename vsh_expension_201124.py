#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: vec_sph_harm.py
"""
Created on Thu May  7 20:22:39 2020

@author: Neo(liuniu@smail.nju.edu.cn)

Calculate the vector spherical harmonics (VSH).

For an introduction of the VSH, please refer to F. Mignard and S. Klioner, A&A 547, A59 (2012).
DOI: 10.1051/0004-6361/201219927.
"""

import numpy as np
from numpy import sqrt, pi, sin, cos, exp
from math import factorial
import sys


# -----------------------------  FUNCTIONS -----------------------------
def B_coef(l, m, x):
    """Calculate the B-function at x.

    The expression of B-function is given in the Eqs.(B.13)-(B.17) in the reference.

    Parameters
    ----------
    l, m: int
        degree and order  of the harmonics
    x: float
        variable
    """

    if type(l) is not int:
        print("The type of 'l' should be int!")
        sys.exit()
    elif type(m) is not int:
        print("The type of 'm' should be int!")
        sys.exit()

    if m == 0:
        return 0
    elif m < 0:
        print("m >= 0")
        sys.exit()

    if l == m:
        if m == 1:
            return 1
        else:
            return (2*m-1) * m / (m-1) * sqrt(1-x*x) * B_coef(m-1, m-1, x)
    elif l - m == 1:
        return (2*m+1) * x * B_coef(m, m, x)
    elif l - m >= 2:
        return ((2*l-1)*x*B_coef(l-1, m, x)-(l-1+m)*B_coef(l-2, m, x)) / (l-m)
    else:
        # For this special case of l < m, return 0 for calculating A_coef
        return 0


def A_coef(l, m, x):
    """Calculate the A-function at x.

    The expression of A-function is given in the Eqs.(B.18)-(B.19) in the reference.

    Parameters
    ----------
    l, m: int
        degree and order  of the harmonics
    x: float
        variable
    """

    if type(l) is not int:
        print("The type of 'l' should be int!")
        sys.exit()
    elif type(m) is not int:
        print("The type of 'm' should be int!")
        sys.exit()

    if m < 0 or l < m:
        print("0 <= m <= l")
        sys.exit()

    if m == 0:
        return sqrt(1-x*x) * B_coef(l, 1, x)
    else:
        return (-x*l*B_coef(l, m, x)+(l+m)*B_coef(l-1, m, x)) / m


def vec_sph_harm_proj(l, m, ra, dc, sph_type="T"):
    """Calculate projection of vsh function of (l,m) at (ra, dc).

    The expression of T_lm and S_lm is given in Eqs.(B.9)-(B.10) in the reference.

    Parameters
    ----------
    l, m: int
        degree and order  of the harmonics
    ra, dc: float
        equatorial coordinates in the unit of radian

    Returns
    -------
    T_ra, T_dc: complex
        Projection of T_lm vector on the e_ra and e_dec vectors.
#   S_ra, S_dc: complex
#       Projection of S_lm vector on the e_ra and e_dec vectors.
    """

    fac = (2*l+1) / (l*l+l) / (4*pi) * factorial(l-m) / factorial(l+m)
    # This expression does not work with array
    # fac = (-1)**m * sqrt(fac) * exp(complex(0, m*ra))
    x, y, z = sin(dc), sin(m*ra), cos(m*ra)
    fac = (-1)**m * sqrt(fac) * (z + y * (0+1j))

    # Projection on ra and decl. direction
    T_ra = fac * A_coef(l, m, x)
    T_dc = fac * B_coef(l, m, x) * (0-1j)

    # Actually S-vector could be derived from T-vector
#    S_ra = facB * (0+1j)
#    S_dc = facA
    if sph_type == "T":
        return T_ra, T_dc
    elif sph_type == "S":
        S_ra = -T_dc
        S_dc = T_ra
        return S_ra, S_dc
    else:
        print("The sph_type can only be 'T' or 'S', Please check it")
        sys.exit()

#    return T_ra, T_dc, S_ra, S_dc


def real_vec_sph_harm_proj(l, m, ra, dc):
    """Calculate the real (not complex) vsh function of (l,m) at x used.

    VSH functions used for real expansion according to Eq.(30) in the reference.

    Please note that the imaginary part has the opposite sign to that of vec_sph_harm.
    And never forget the factor of 2.

    Parameters
    ----------
    l, m: int
        degree and order of the harmonics
    ra, dc: float
        equatorial coordinates in the unit of radian

    Returns
    -------
    T_ra, T_dc: complex
        Projection of T_lm vector on the e_ra and e_dec vectors.
#    S_ra, S_dc: complex
#        Projection of S_lm vector on the e_ra and e_dec vectors.
    """

#    T_ra, T_dc, S_ra, S_dc = vec_sph_harm(l, m, ra, dc)
    T_ra, T_dc = vec_sph_harm_proj(l, m, ra, dc)
#    Maybe this step is unneccessary
#    S_ra, S_dc = -T_dc, T_ra

    if m:
        T_ra_r, T_ra_i = 2*np.real(T_ra), -2*np.imag(T_ra)
        T_dc_r, T_dc_i = 2*np.real(T_dc), -2*np.imag(T_dc)
        # To avoid repeated calculation
        # S_ra_r, S_ra_i = np.real(S_ra), -np.imag(S_ra)
        # S_dc_r, S_dc_i = np.real(S_dc), -np.imag(S_dc)

#        return T_ra_r, T_dc_r, T_ra_i, T_dc_i, S_ra_r, S_dc_r, S_ra_i, S_dc_i
        return T_ra_r, T_dc_r, T_ra_i, T_dc_i
    else:
        #        return T_ra, T_dc, S_ra, S_dc
        return np.real(T_ra), np.real(T_dc)
# --------------------------------- END --------------------------------
