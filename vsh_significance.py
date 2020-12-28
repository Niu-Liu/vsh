#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: vsh_significance.py
"""
Created on Fri Nov 27 19:57:26 2020

@author: Neo(niu.liu@nju.edu.cn)

Check the significance level of VSH
"""

import sys
import numpy as np
from numpy import sqrt
# My progs
from .vsh_power import calc_vsh_power_lm, calc_vsh_power, calc_rem_power, calc_field_power


def calc_vsh_nor_power_lm1(ts_lm, ts_lm_err, l):
    """Calculate normalized power#1 of toroidal/spheroidal components of degree l

    This quantity is given by Eqs.(83)

    Parameters
    ----------
    ts_lm : array of (2*l+1)
        coefficient of t_lm or s_lm, m=1,2,..,l
    ts_l_errm : array of (2*l+1)
        standard deviation of coefficient of t_lm or s_lm, m=1,2,..,l
    l : int
        degree

    Return
    ------
    Wts_l1: float
        normalized powers
    """

    # Check of ts_lm, ts_lm_err, and l match
    if len(ts_lm) != 2*l+1:
        print("The length of ts_lm and l does not matchi!")
        sys.exit()
    if len(ts_lm) != len(ts_lm_err):
        print("The length of ts_lm and ts_lm_err should be equal!")
        sys.exit()

    # Calculate the normal power
    Pts_l = calc_vsh_power_lm(ts_lm, l)
    Wts_li = Pts_l / ts_lm_err[0]**2

    return Wts_l1


def calc_vsh_nor_power_lm2(ts_lm, ts_lm_err, l):
    """Calculate normalized power#2 of toroidal/spheroidal components of degree l

    This quantity is given by Eqs.(86)

    Parameters
    ----------
    ts_lm : array of (2*l+1)
        coefficient of t_lm or s_lm, m=1,2,..,l
    ts_l_errm : array of (2*l+1)
        standard deviation of coefficient of t_lm or s_lm, m=1,2,..,l
    l : int
        degree

    Return
    ------
    Wts_l2: float
        normalized powers
    """

    # Check of ts_lm, ts_lm_err, and l match
    if len(ts_lm) != 2*l+1:
        print("The length of ts_lm and l does not matchi!")
        sys.exit()
    if len(ts_lm) != len(ts_lm_err):
        print("The length of ts_lm and ts_lm_err should be equal!")
        sys.exit()

    # Calculate the normal power
    Wts_l2 = np.sum((t_lm/t_lm_err)**2)

    return Wts_l2


def calc_vsh_nor_power_lm3(ts_lm, ts_lm_err, l):
    """Calculate normalized power#3 of toroidal/spheroidal components of degree l

    This quantity is given by Eqs.(87)

    Parameters
    ----------
    ts_lm : array of (2*l+1)
        coefficient of t_lm or s_lm, m=1,2,..,l
    ts_l_errm : array of (2*l+1)
        standard deviation of coefficient of t_lm or s_lm, m=1,2,..,l
    l : int
        degree

    Return
    ------
    Wts_l3: float
        normalized powers
    """

    # Check of ts_lm, ts_lm_err, and l match
    if len(ts_lm) != 2*l+1:
        print("The length of ts_lm and l does not matchi!")
        sys.exit()
    if len(ts_lm) != len(ts_lm_err):
        print("The length of ts_lm and ts_lm_err should be equal!")
        sys.exit()

    # Calculate the normal power
    Pts_l = calc_vsh_power_lm(ts_lm, l)
    ts_lm_err2 = ts_lm_err**2
    mean_sigma2 = (np.sum(ts_lm_err2)*2 - ts_lm_err2[0]) / (2*l+1)
    Wts_l3 = Pts_l / mean_sigma2

    return Wts_l3


def calc_vsh_nor_power_l(ts_pmt, ts_pmt_err, l_max, nor_type="3"):
    """Calculate normalized power for all degree l=1,2,...,l_max

    Parameters
    ----------
    ts_pmt : array of (l_max+2)*l_max
        coefficient of t_lm or s_lm, m=1,2,..,l, l=1,2,...,l_max
    ts_pmt_err : array of (l_max+2)*l_max
        standard deviation of coefficient of t_lm or s_lm, m=1,2,..,l, l=1,2,...,l_max
    l_max : int
        maximum degree

    Return
    ------
    Wts: float
        normalized powers at every degree l=1,2,...,l_max
    """

    if nor_type not in ["1", "2", "3"]:
        print("nor_type should only be 1, 2, or 3")
        sys.exit()

    # Check of ts_pmt and l_max match
    if len(ts_pmt) != (l_max+2)*l_max:
        print("The length of ts_pmt and l_max does not matchi!")
        sys.exit()
    if len(ts_pmt) != len(ts_pmt_err):
        print("The length of ts_pmt and ts_pmt_err should be equal!")
        sys.exit()

    # loop of l=1:l_max+1
    Wts = np.zeros(l_max)
    end = 0
    for l in range(1, l_max+1):
        sta, end = end, end + (l*2+1)
        ts_lm = ts_pmt[sta:end]
        ts_lm_err = ts_pmt_err[sta:end]

        # First type
        if nor_type == "1":
            Wts[l-1] = calc_vsh_nor_power_lm1(ts_lm, ts_lm_err, l)
        elif nor_type == "2":
            Wts[l-1] = calc_vsh_nor_power_lm2(ts_lm, ts_lm_err, l)
        else:
            Wts[l-1] = calc_vsh_nor_power_lm3(ts_lm, ts_lm_err, l)

    return Wts


def calc_vsh_nor_power(ts_pmt, ts_pmt_err, l_max, nor_type="3"):
    """Calculate normalized power of toroidal and spheroidal component

    Parameters
    ----------
    ts_pmt : array of (l_max+2)*l_max*2
        coefficient of t_lm and s_lm, m=1,2,..,l, l=1,2,...,l_max
    ts_pmt_err : array of (l_max+2)*l_max
        standard deviation of coefficient of t_lm or s_lm, m=1,2,..,l, l=1,2,...,l_max
    l_max : int
        maximum degree

    Return
    ------
    W_t/W_s: array of (l_max,1)
        normalized powers of toroidal/spheroidal
    """

    # Check of ts_pmt and l_max match
    if len(ts_pmt) != 2*(l_max+2)*l_max:
        print("The length of ts_pmt and l_max does not matchi!")
        sys.exit()
    if len(ts_pmt) != len(ts_pmt_err):
        print("The length of ts_pmt and ts_pmt_err should be equal!")
        sys.exit()

    # Separate t_lm and s_lm
    t_pmt, t_pmt_err = ts_pmt[0::2], ts_pmt_err[0::2]
    s_pmt, s_pmt_err = ts_pmt[1::2], ts_pmt_err[1::2]

    # Normalized power of toroidal and spheroidal component
    W_t = calc_vsh_nor_power_l(t_pmt, t_pmt_err, l_max, nor_type)
    W_s = calc_vsh_nor_power_l(s_pmt, s_pmt_err, l_max, nor_type)

    return W_t, W_s


def calc_Z_lm(Wts_l, num_dof):
    """Calculate Z-statistics in Eq. (85) at degree l

    Parameters
        W-statistics at degree l
    num_dof : int
        number of degrees of freedom

    Return
    ------
    Zts_l : float
        Z-statistics at degree l
    """

    Zts_l = sqrt(4.5*num_dof) * ((Wts_l/num_dof)**(1./3)-(1-2/(9*num_dof)))

    return Zts_l


def calc_Z(Wts):
    """Calculate Z-statistics in Eq. (85) for each degree l

    Parameters
    ----------
    Wts : float
        W-statistics
#    num_dof : int
#        number of degrees of freedom

    Return
    ------
    Zts : float
        Z-statistics
    """

    Zts = np.zeros_like(Wts)

    for l, Wts_l in enumerate(Wts):
        Zts[l] = calc_Z_lm(Wts_l, 2*l+1)

    return Zts


def check_vsh_sig(dra, ddc, ts_pmt, ts_pmt_err, l_max):
    """Check significance of VSH terms

        standard deviation of coefficient of t_lm or s_lm, m=1,2,..,l, l=1,2,...,l_max
    l_max : int
        maximum degree
#    num_dof : int
#        number of degrees of freedom
    """

    # Power of vector field
    P_field = calc_field_power(dra, ddc)

    # Power of degree l
    P_t, P_s = calc_vsh_power(ts_pmt, l_max)

    # Remaining power
    P_rem = calc_rem_power(P_t+P_s, P_field, l_max)

    # Omega-statistics
    W_t, W_s = calc_vsh_nor_power(ts_pmt, ts_pmt_err, l_max)

    # Z-statistics
    Z_t, Z_s = calc_Z(W_t), calc_Z(W_s)

    # Print these information
    print("Significance level ")
    print("--------------------------------------------------------------------")
    print("Degree  P_t^0.5      Z_t   P_s^0.5      Z_s  P_rem^0.5")
    print("--------------------------------------------------------------------")

    # Total powers
    print("{:6d}                                          "
          "{:6.1f}".format(0, sqrt(P_rem[0])))
    for l in range(0, l_max):
        print("{:6d} {:8.1f}   {:6.2f}  {:8.1f}   {:6.2f}  {:9.1f}".format(
            l+1, sqrt(P_t[l]), Z_t[l], sqrt(P_s[l]), Z_s[l], sqrt(P_rem[l+1])))

    print("--------------------------------------------------------------------")


def main():
    """A simple test"""

    l_max = 15
    ts_pmt = np.random.random(2*l_max*(l_max+2)) * 100
    ts_pmt_err = np.fabs(np.random.random(2*l_max*(l_max+2)) * 30)
    num_dof = 1e5

    check_vsh_sig(ts_pmt, ts_pmt_err, l_max, num_dof)


if __name__ == "__main__":
    main()
# --------------------------------- END --------------------------------
