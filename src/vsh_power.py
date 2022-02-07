#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: vsh_power.py
"""
Created on Fri Nov 27 19:17:24 2020

@author: Neo(niu.liu@nju.edu.cn)

Calculate the power of vsh terms
"""

import sys
import numpy as np


# -----------------------------  MAIN -----------------------------
def calc_vsh_power_lm(ts_lm, l):
    """Calculate power of toroidal/spheroidal components of degree l

    Parameters
    ----------
    ts_lm : array of (2*l+1)
        coefficient of t_lm or s_lm, m=1,2,..,l
    l : int
        degree

    Return
    ------
    Pts_l: float
        powers
    """

    # Check of ts_lm and l match
    if len(ts_lm) != 2*l+1:
        print("The length of ts_lm and l does not match!")
        sys.exit()

    # Calculate the power
    ts_lm2 = ts_lm**2
    Pts_l = 2 * np.sum(ts_lm2) - ts_lm2[0]

    return Pts_l


def calc_vsh_power_l(ts_pmt, l_max):
    """Calculate power of toroidal/spheroidal component of all degree

    Parameters
    ----------
    ts_pmt : array of (l_max+2)*l_max
        coefficient of t_lm or s_lm, m=1,2,..,l, l=1,2,...,l_max
    l_max : int
        maximum degree

    Return
    ------
    Pts: float
        powers at every degree l=1,2,...,l_max
    """

    # Check of ts_pmt and l_max match
    if len(ts_pmt) != (l_max+2)*l_max:
        print("The length of ts_pmt and l_max does not match!")
        sys.exit()

    # loop of l=1:l_max+1
    Pts = np.zeros(l_max)
    end = 0
    for l in range(1, l_max+1):
        sta, end = end, end + (l*2+1)
        ts_lm = ts_pmt[sta:end]
        Pts[l-1] = calc_vsh_power_lm(ts_lm, l)

    return Pts


def calc_vsh_power(ts_pmt, l_max, fit_type="Full"):
    """Calculate power of toroidal and spheroidal component

    Parameters
    ----------
    ts_pmt : array of (l_max+2)*l_max*2
        coefficient of t_lm and s_lm, m=1,2,..,l, l=1,2,...,l_max
    l_max : int
        maximum degree

    Return
    ------
    P_t/P_s: array of (l_max,1)
        powers of toroidal/spheroidal
    """

    if fit_type in ["FULL", "full", "f", "F"]:
        N = 2*(l_max+2)*l_max
    elif fit_type in ["S", "T", "s", "t"]:
        N = (l_max+2)*l_max
    else:
        print("Unrecognized fit_type!")
        sys.exit()

    # Check of ts_pmt and l_max match
    if len(ts_pmt) != N:
        print("The length of ts_pmt and l_max does not match!")
        sys.exit()

    if fit_type in ["FULL", "full", "f", "F"]:
        # Both S and T vector
        # Separate t_lm and s_lm
        t_pmt = ts_pmt[0::2]
        s_pmt = ts_pmt[1::2]
    elif fit_type in ["T", "t"]:
        t_pmt = ts_pmt
        s_pmt = np.zeros_like(t_pmt)
    else:
        t_pmt = np.zeros_like(t_pmt)
        s_pmt = ts_pmt

    # Power of toroidal and spheroidal component
    P_t = calc_vsh_power_l(t_pmt, l_max)
    P_s = calc_vsh_power_l(s_pmt, l_max)

    # Total power
#    P_tol = P_t + P_s
    return P_t, P_s


def calc_field_power(dra, ddc):
    """Calculate power of vector field according to Eq.(74)

    Parameters
    ----------
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences, mostly in dex

    Return
    ------
    P_field : float
        power of the vector field
    """

    P_field = 4 * np.pi / len(dra) * np.sum(dra**2 + ddc**2)

    return P_field


def calc_rem_power(P_tot, P_field, l_max):
    """Calculate the power in the remaining signal

    Parameters
    ----------
    P_tot : array of (l_max,1)
        total power at each degree l=1,2,...,l_max
    l_max : int
        maximum degree
    P_field : float
        power of the vector field

    Return
    ------
    P_rem : array of (l_max+1,1)
        remaining power, first element being the total powers of the field
    """

    P_rem = np.zeros(l_max+1)
    P_rem[0] = P_field

    for l in range(1, l_max+1):
        P_rem[l] = P_rem[l-1] - P_tot[l-1]

    return P_rem
# --------------------------------- END --------------------------------
