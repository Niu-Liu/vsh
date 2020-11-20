#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: test.py
"""
Created on Wed Nov 18 11:19:47 2020

@author: Neo(niu.liu@nju.edu.cn)
"""

import numpy as np


# -----------------------------  MAIN -----------------------------
from astropy.table import Table
import numpy as np

# My modules
from vsh_fit import vsh_fit
from pmt_convert import st_to_rotgld, st_to_rotgldquad


test_tab = Table.read("test.csv")

# Transform astropy.Column into np.array and mas -> uas
dra = np.array(test_tab["dra"] * 1e3)
ddec = np.array(test_tab["ddec"] * 1e3)
dra_err = np.array(test_tab["dra_err"] * 1e3)
ddec_err = np.array(test_tab["ddec_err"] * 1e3)
ra = np.array(test_tab["ra"])
dec = np.array(test_tab["dec"])
dra_ddec_cov = np.array(test_tab["dra_ddec_cov"] * 1e6)

print("        "
      "               Rotation [uas]             "
      "               Glide [uas]")
print("        "
      "     R1            R2            R3        "
      "     G1            G2            G3         ")
print("        "
      "-----------------------------------------   "
      "-----------------------------------------")

for l in range(1, 6):
    pmt, err, cor_mat = vsh_fit(
        dra, ddec, dra_err, ddec_err, ra, dec, dra_ddec_cov, l_max=l)

    pmt1, err1, cor_mat1 = st_to_rotgld(pmt[:6], err[:6], cor_mat[:6, :6])

    print("l_max={12:2d} "
          "{0:4.0f} +/- {6:4.0f} {1:4.0f} +/- {7:4.0f} "
          "{2:4.0f} +/- {8:4.0f} {3:4.0f} +/- {9:4.0f} "
          "{4:4.0f} +/- {10:4.0f} {5:4.0f} +/- {11:4.0f}  ".format(*pmt1, *err1, l))


# --------------------------------- END --------------------------------
