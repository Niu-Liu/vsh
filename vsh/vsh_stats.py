#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: vsh_stat.py
"""
Created on Sat Nov 28 13:04:20 2020

@author: Neo(niu.liu@nju.edu.cn)
"""

import numpy as np
from .stats_func import (calc_mean, calc_wrms,
                         calc_rse, calc_chi2_2d, calc_gof)


# -----------------------------  MAIN -----------------------------
def residual_stats_calc(dra, ddc, dra_err, ddc_err, ra_dc_cor, num_dof):
    """Calculate statistics for the residuals

    Parameters
    ----------
    dra/ddc : array of float
        R.A.(*cos(Dec.))/Dec. differences in dex
    dra_err/ddc_err : array of float
        formal uncertainty of dra(*cos(dc_rad))/ddc in dex
    ra_rad/dc_rad : array of float
        Right ascension/Declination in radian
    ra_dc_cor : array of float
        correlation coefficient between dra and ddc, default is None
    num_dof : int
        number of degree of freedom

    Return
    ------
    stats : dict
        containing all the statstics
    """

    stats = {}

    # Offset of RA/Dec residual
    stats["dra_bias"] = calc_mean(dra, dra_err)
    stats["ddec_bias"] = calc_mean(ddc, ddc_err)

    # WRMS of RA/Dec residual
    stats["dra_wrms"] = calc_wrms(dra, dra_err, stats["dra_bias"])
    stats["ddec_wrms"] = calc_wrms(ddc, ddc_err, stats["dra_bias"])

    # RSE of RA/Dec residual
    stats["dra_rse"] = calc_rse(dra)
    stats["ddec_rse"] = calc_rse(ddc)

    # Normalized residual
    nor_dra = dra / dra_err
    nor_ddc = ddc / ddc_err

    # Offset of normalized RA/Dec residual
    stats["nor_dra_bias"] = calc_mean(nor_dra)
    stats["nor_ddec_bias"] = calc_mean(nor_ddc)

    # Standard deviation of RA/Dec residual
    stats["nor_dra_std"] = calc_wrms(nor_dra)
    stats["nor_ddec_std"] = calc_wrms(nor_ddc)

    # RSE of normalized RA/Dec residual
    stats["nor_dra_rse"] = calc_rse(nor_dra)
    stats["nor_ddec_rse"] = calc_rse(nor_ddc)

    # Calculate the reduced Chi-square
    stats["chi2"] = calc_chi2_2d(
        dra, dra_err, ddc, ddc_err, ra_dc_cor)
    stats["reduced_chi2"] = stats["chi2"] / num_dof

    # Calculate the goodness-of-fit
    stats["gof"] = calc_gof(num_dof, stats["chi2"])

    return stats


def print_stats_info(apr_stats, pos_stats):
    """Print statistics information for comparison

    Parameters
    ----------
    apr_stats/pos_stats : dict
        containing all the statistics for pre-fit/post-fit data
    """

    print("")
    print("Statistics information")
    print("-------------------------------------------------------------------")
    print("             Pre-fit             |             Post-fit            ")
    print("-------------------------------------------------------------------")
    print("         dRA*         dDec       |         dRA*         dDec       ")
    print("Bias  {:7.1f}      {:7.1f}       |"
          "      {:7.1f}      {:7.1f}".format(apr_stats["dra_bias"], apr_stats["ddec_bias"],
                                              pos_stats["dra_bias"], pos_stats["ddec_bias"]))
    print("WRMS  {:7.1f}      {:7.1f}       |"
          "      {:7.1f}      {:7.1f}".format(apr_stats["dra_wrms"], apr_stats["ddec_wrms"],
                                              pos_stats["dra_wrms"], pos_stats["ddec_wrms"]))
    print("RSE   {:7.1f}      {:7.1f}       |"
          "      {:7.1f}      {:7.1f}".format(apr_stats["dra_rse"], apr_stats["ddec_rse"],
                                              pos_stats["dra_rse"], pos_stats["ddec_rse"]))
    print("-------------------------------------------------------------------")
    print("    Nor. dRA*    Nor. dDec       |    Nor. dRA*    Nor. dDec       ")
    print("-------------------------------------------------------------------")
    print("         dRA*         dDec       |         dRA*         dDec       ")
    print("Bias  {:7.1f}      {:7.1f}       |"
          "      {:7.1f}      {:7.1f}".format(apr_stats["nor_dra_bias"], apr_stats["nor_ddec_bias"],
                                              pos_stats["nor_dra_bias"], pos_stats["nor_ddec_bias"]))
    print("Std   {:7.1f}      {:7.1f}       |"
          "      {:7.1f}      {:7.1f}".format(apr_stats["nor_dra_std"], apr_stats["nor_ddec_std"],
                                              pos_stats["nor_dra_std"], pos_stats["nor_ddec_std"]))
    print("RSE   {:7.1f}      {:7.1f}       |"
          "      {:7.1f}      {:7.1f}".format(apr_stats["nor_dra_rse"], apr_stats["nor_ddec_rse"],
                                              pos_stats["nor_dra_rse"], pos_stats["nor_ddec_rse"]))
    print("-------------------------------------------------------------------")
    print("Chi-Square       {:9.1f}       |"
          "                 {:9.1f}".format(apr_stats["chi2"], pos_stats["chi2"]))
    print("Red. Chi-square  {:9.1f}       |"
          "                 {:9.1f}".format(apr_stats["reduced_chi2"], pos_stats["reduced_chi2"]))
    print("Good-of-fitness  {:9.1f}       |"
          "                 {:9.1f}".format(apr_stats["gof"], pos_stats["gof"]))
    print("-------------------------------------------------------------------")
    print("")

# --------------------------------- END --------------------------------
