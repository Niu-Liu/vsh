#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: vsh_aux_info.py
"""
Created on Sat Nov 28 12:20:02 2020

@author: Neo(niu.liu@nju.edu.cn)
"""

import numpy as np


# -----------------------------  MAIN -----------------------------
def prefit_check(num_sou, num_pmt):
    """Check if conditions are all met

    Parameters
    ----------
    num_sou: int
        number of sources
    num_pmt : int
        number of unknowns
    """

    # Pre-fit check
    if 2 * num_sou < num_pmt:
        print("The sample of {} sources is insufficient for determining "
              "{:d} unknowns!".format(num_sou, num_pmt))
        print("Abort!")
        sys.exit(1)


def prefit_info(num_sou, num_pmt, num_dof, l_max):
    """Print prefit information

    Parameters
    ----------
    num_sou: int
        number of sources
    num_pmt : int
        number of unknowns
    """

    print("")
    print("Pre-fit information:")
    print("    Number of sources                {:10d}".format(num_sou))
    print("    Maximum degree of VSH            {:10d}".format(l_max))
    print("    Number of unknowns to determine  {:10d}".format(num_pmt))
    print("    Number of degree of freedom      {:10d}".format(num_dof))
    print("")
# --------------------------------- END --------------------------------
