#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
# File name: ssh_expansion.py
"""
Created on Tue Oct  5 19:52:34 2021

@author: Neo(niu.liu@nju.edu.cn)

This script contains functions for calculationg the scalar spherical harmonics.

"""


import sys

import numpy as np
from numpy import sqrt, pi, sin, cos, exp
from math import factorial


# -----------------------------  FUNCTIONS -----------------------------
def sca_sph_harm_proj(l_max, ra, dc):
    """Calculate the SSH function at (ra, dec)

    Parameters
    ----------
    l_max: int
        maximum degree of the harmonics
    ra, dc: float
        equatorial coordinates in the unit of radian

    Returns
    -------
    Y_mat: complex
    """

    # In some cases that l_max is not an integer
    l_max = int(l_max)

    # Calculate P_mn(x)
    x = sin(dc)
    num_sou = len(ra)
    fac = np.sqrt(1-x*x)

    # Initialize the Pmn
    P_mat = np.zeros((l_max+1, l_max+1, num_sou))
    Y_mat = np.zeros((l_max+1, l_max+1, num_sou)) * (0+0j)

    # Generate the sequence of Pmn, c.f. Eqs. (B.4-B.7)
    P_mat[0, 0, :] = 1
    for l in range(1, l_max+1):
        for m in range(l+1)[::-1]:
            if m:
                if l == m:
                    P_mat[l, m, :] = fac * (2*m-1) * P_mat[m-1, m-1, :]
                elif l == m+1:
                    P_mat[l, m, :] = (2*m+1) * x * P_mat[m, m, :]
                else:
                    P_mat[l, m, :] = ((2*l-1)*x*P_mat[l-1, m, :] -
                                      (l-1+m)*P_mat[l-2, m, :]) / (l-m)
            # else:
                # P_mat[l, m, :] = 0

    # Calculate the spherical harmonics
    for l in range(1, l_max+1):
        for m in range(l+1):
            # Calculate the coefficient in Eqs. (9-10)
            fac = (2*l+1) / (l*l+l) / (4*pi) * factorial(l-m) / factorial(l+m)
            fac = (-1)**m * sqrt(fac) * (cos(m*ra)+sin(m*ra)*(0+1j))

            Y_mat[l, m, :] = fac * P_mat[l, m, :]

    return Y_mat


def real_sca_sph_harm_proj(l, m, Y_mat):
    """Calculate the real (not complex) SSH function of (l,m) at x used.

    SSH functions used for real expansion are analogous to the VSH.

    Please note that the imaginary part has the opposite sign to that of vec_sph_harm.
    And never forget the factor of 2.

    Parameters
    ----------
    l, m: int
        degree and order of the harmonics

    Returns
    -------
    Y_r/i: real vector
        Real and imaginary part of the harmonics
    """

    Y = Y_mat[l, m, :]

    if m:
        Y_r, Y_i = 2*np.real(Y_mat), -2*np.imag(Y_mat)
        return Y_r, Y_i
    else:
        return Y
