#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: stats_calc.py
"""
Created on Wed Feb 14 11:02:00 2018

@author: Neo(liuniu@smail.nju.edu.cn)

This script is used for calculating the pre-fit wrms,
post-fit wrms, reduced-chi square, and standard deviation.

3 Mar 2018, Niu : now function "calc_wrms" also computes the standard
                  deviation

"""

import sys
from functools import reduce
import numpy as np
from scipy.special import gammaincc

__all__ = ["calc_mean", "calc_wrms", "calc_chi2",
           "calc_chi2_2d", "calc_gof", "calc_rse"]


# -----------------------------  FUNCTIONS -----------------------------
def calc_mean(x, err=None):
    """Calculate the mean value.

    Parameters
    ----------
    x : array, float
        Series
    err : array, float, default is None.

    Returns
    ----------
    mean : float
        mean
    """

    if err is None:
        mean = np.mean(x)
    else:
        p = 1. / err
        mean = np.dot(x*p, p) / np.dot(p, p)

    return mean


def calc_wrms(x, err=None, bias=None):
    """Calculate the (weighted) wrms of x series after removing
    the bias.

    Standard deviation
    std = sqrt(sum( (xi-mean)^2/erri^2 ) / sum( 1.0/erri^2 ))
         if weighted,
         = sqrt(sum( (xi-mean)^2/erri^2 ) / (N-1))
         otherwise.

    Weighted root mean square
    wrms = sqrt(sum( xi^2/erri^2 ) / sum( 1.0/erri^2 ))
         if weighted,
         = sqrt(sum( xi^2/erri^2 ) / (N-1))
         otherwise.

    When bias is 0, calc_wrms is equivalent to calc_wrms2

    Parameters
    ----------
    x : array, float
        Series
    err : array, float, default is None.

    Returns
    ----------
    wrms : float
        weighted rms
    """

    if err is None:
        if bias is None:
            bias = np.mean(x)
            xn = x - bias
            wgt = len(x) - 1
    else:
        wgt = 1. / err
        bias = np.dot(x, wgt**2) / np.dot(wgt, wgt)
        xn = (x - bias) * wgt
        wgt = np.dot(wgt, wgt)

    wrms = np.sqrt(np.dot(xn, xn) / wgt)

    return wrms


def calc_chi2(x, err, reduced=False, num_dof=None):
    """Calculate the (reduced) Chi-square.


    Parameters
    ----------
    x : array, float
        residuals
    err : array, float
        formal errors of residuals
    reduced : boolean
        True for calculating the reduced chi-square
    num_dof: int
        number of degree of freedom

    Returns
    ----------
    (reduced) chi-square
    """

    wgt_x = x / err
    chi2 = np.dot(wgt_x, wgt_x)

    if reduced:
        if num_dof is None:
            print("Please provide the number of degree of freedom")
            sys.exit(1)

        chi2 = chi2 / num_dof

    return chi2


def calc_chi2_2d(x, x_err, y, y_err, xy_cov=None, xy_cor=None, reduced=False, num_dof=None):
    """Calculate the 2-Dimension (reduced) Chi-square.


    Parameters
    ----------
    x : array, float
        residuals of x
    x_err : array, float
        formal errors of x
    y : array, float
        residuals of x
    y_err : array, float
        formal errors of x
    xy_cov : array, float
        covariance between x and y
    xy_cor : array, float
        correlation coefficient between x and y
    reduced : boolean
        True for calculating the reduced chi-square
    num_dof: int
        number of degree of freedom

    Returns
    ----------
    (reduced) chi-square
    """

    chi2_array = np.zeros_like(x)

    if xy_cov is None:
        if xy_cor is not None:
            xy_cov = xy_cor * x_err * y_err
        else:
            xy_cov = np.zeros_like(x)

    for i, (xi, x_erri, yi, y_erri, xy_covi) in enumerate(
            zip(x, x_err, y, y_err, xy_cov)):

        wgt_mat = np.linalg.inv(np.array([[x_erri**2, xy_covi],
                                          [xy_covi, y_erri**2]]))

        xy_mat = np.array([xi, yi])

        chi2_array[i] = reduce(np.dot, (xy_mat, wgt_mat, xy_mat))

    chi2 = np.sum(chi2_array)

    if reduced:
        if num_dof is None:
            print("Please provide the number of degree of freedom")
            sys.exit(1)

        chi2 = chi2 / num_dof

    return chi2


def calc_gof(num_dof, chi2):
    """Calculate the goodness-of-fit.

    The formula is expressed as below.

    Q = gammq(num_dof / 2, chi2 / 2). (Numerical Recipes)

    gammq is the incomplete gamma function.

    Parameters
    ----------
    num_dof : int
        number of freedom
    chi2 : float
        chi-square

    Return
    ------
    gof : float
        goodness-of-fit
    """

    gof = gammaincc(num_dof/2, chi2/2)

    return gof


def calc_rse(x):
    """Calculate Robust Scatter Estimate (RSE) estimator

    RSE is used as a standardized, robust measure of dispersion in the Gaia group.
    """

    x10, x90 = np.percentile(x, [10, 90])
    rse = 0.390152 * (x90 - x10)

    return rse
# --------------------------------- END --------------------------------
