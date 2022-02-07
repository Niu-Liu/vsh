#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: rgq_func.py
"""
Created on Tue Nov 24 13:23:41 2020

@author: Neo(niu.liu@nju.edu.cn)

VSH of degree two in terms of rotation, glide, and quadrupolar terms
"""

import sys
from numpy import sin, cos


# -----------------------------  MAIN -----------------------------
def vsh_func01(ra, dec, g1, g2, g3, r1, r2, r3):
    """VSH function of the first degree.

    Parameters
    ----------
    ra/dec : array of float
        Right ascension/Declination in radian
    r1, r2, r3 : float
        rotation parameters
    g1, g2, g3 : float
        glide parameters

    Returns
    ----------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences in dex
    """

    dra = [-r1 * cos(ra) * sin(dec) - r2 * sin(ra) * sin(dec)
           + r3 * cos(dec)
           - g1 * sin(ra) + g2 * cos(ra)][0]
    ddec = [+ r1 * sin(ra) - r2 * cos(ra)
            - g1 * cos(ra) * sin(dec) - g2 * sin(ra) * sin(dec)
            + g3 * cos(dec)][0]

    return dra, ddec


def vsh_func02(ra, dec,
               ER_22, EI_22, ER_21, EI_21, E_20,
               MR_22, MI_22, MR_21, MI_21, M_20):
    """VSH function of the second degree.

    Parameters
    ----------
    ra/dec : array of float
        Right ascension/Declination in radian
    E_20, ER_21, EI_21, ER_22, EI_22 : float
        quadrupolar parameters of electric type
    M_20, MR_21, MI_21, MR_22, MI_22 : float
        quadrupolar parameters of magnetic type

    Returns
    ----------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences in dex
    """

    dra = [+M_20 * sin(2 * dec)
           - (MR_21 * cos(ra) - MI_21 * sin(ra)) * cos(2*dec)
           + (ER_21 * sin(ra) + EI_21 * cos(ra)) * sin(dec)
           - (MR_22 * cos(2*ra) - MI_22 * sin(2*ra)) * sin(2*dec)
           - 2*(ER_22 * sin(2*ra) + EI_22 * cos(2*ra)) * cos(dec)][0]
    ddec = [+E_20 * sin(2 * dec)
            - (MR_21 * sin(ra) + MI_21 * cos(ra)) * sin(dec)
            - (ER_21 * cos(ra) - EI_21 * sin(ra)) * cos(2*dec)
            + 2*(MR_22 * sin(2*ra) + MI_22 * cos(2*ra)) * cos(dec)
            - (ER_22 * cos(2*ra) - EI_22 * sin(2*ra)) * sin(2*dec)][0]

    return dra, ddec


def vsh_func(ra, dec, pmt, l_max=2):
    """VSH function of degree 2.

    Parameters
    ----------
    ra/dec : array of float
        Right ascension/Declination in radian
    pmt : array
        parameters
    l_max : int
        maximum degree

    Returns
    ----------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences in dex
    """

    if l_max == 1:
        dra, ddec = vsh_func01(ra, dec, *pmt)
    elif l_max == 2:
        dra1, ddec1 = vsh_func01(ra, dec, *pmt[:6])
        dra2, ddec2 = vsh_func02(ra, dec, *pmt[6:])
        dra, ddec = dra1 + dra2, ddec1 + ddec2
    else:
        print("Not supported!")
        sys.exit()

    return dra, ddec


# --------------------------------- END --------------------------------
