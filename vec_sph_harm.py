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
def B_func(l, m, x):
    """Calculate the B-function at x.

    The expression of B-function is given in the Eqs.(B.13)-(B.17) in the reference.

    Parameters
    ----------
    l, m: int
        degrees of the harmonics
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

    if l == m:
        if m == 1:
            return 1
        else:
            return (2*m - 1) * m / (m - 1) * sqrt(1-x*x) * B_func(m-1, m-1, x)

    elif l - m == 1:
        return (2*m + 1) * x * B_func(m, m, x)

    else:
        return ((2*l-1)*x*B_func(l-1, m, x) - (l-1+m)*B_func(l-2, m, x)) / (l-m)


def A_func(l, m, x):
    """Calculate the A-function at x.

    The expression of A-function is given in the Eqs.(B.18)-(B.19) in the reference.

    Parameters
    ----------
    l, m: int
        degrees of the harmonics
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
        return sqrt(1 - x*x) * B_func(l, 1, x)
    else:
        return (-x * l * B_func(l, m, x) + (l+m) * B_func(l-1, m, x)) / m


def vec_sph_harm(l, m, ra, dec):
    """Calculate the vsh function of (l,m) at x.

    The expression of T_lm and S_lm is given in Eqs.(B.9)-(B.10) in the reference.

    Parameters
    ----------
    l, m: int
        degrees of the harmonics
    ra, dec: float
        equatorial coordinates in the unit of radian

    Returns
    -------
    T_ra, T_dc: complex
        Projection of T_lm vector on the e_ra and e_dec vectors.
    S_ra, S_dc: complex
        Projection of S_lm vector on the e_ra and e_dec vectors.
    """

    fac1 = (2*l+1) / l / (l+1) / (4 * pi) * factorial(l-m) / factorial(l+m)
    fac2 = (-1)**m * sqrt(fac1) * exp(complex(0, m*ra))

    x = sin(dec)

    facA = fac2 * A_coef(l, m, x)
    facB = fac2 * B_coef(l, m, x)

    T_ra = facA
    T_dc = facB * (0-1j)

    S_ra = facB * (0+1j)
    S_dc = facA

    return T_ra, T_dc, S_ra, S_dc


def real_vec_sph_harm(l, m, ra, dec):
    """Calculate the real (not complex) vsh function of (l,m) at x used.

    VSH functions used for real expansion according to Eq.(30) in the reference.

    For m=0, real_vec_sph_harm is equivlent to vec_sph_harm.

    Please note that the imaginary part has the opposite sign to that of vec_sph_harm.

    Parameters
    ----------
    l, m: int
        degrees of the harmonics
    ra, dec: float
        equatorial coordinates in the unit of radian

    Returns
    -------
    T_ra, T_dc: complex
        Projection of T_lm vector on the e_ra and e_dec vectors.
    S_ra, S_dc: complex
        Projection of S_lm vector on the e_ra and e_dec vectors.
    """

    T_ra, T_dc, S_ra, S_dc = vec_sph_harm(l, m, ra, dec)

    if m:
        T_ra_R, T_ra_i = np.real(T_ra), -np.imag(T_ra)
        T_dc_R, T_dc_i = np.real(T_dc), -np.imag(T_dc)
        S_ra_R, S_ra_i = np.real(S_ra), -np.imag(S_ra)
        S_dc_R, S_dc_i = np.real(S_dc), -np.imag(S_dc)

        return T_ra_R, T_dc_R, T_ra_i, T_dc_i, S_ra_R, S_dc_R, S_ra_i, S_dc_i
    else:
        return T_ra, T_dc, S_ra, S_dc
# --------------------------------- END --------------------------------
