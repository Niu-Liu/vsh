#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: test.py
"""
Created on Wed Nov 18 11:19:47 2020

@author: Neo(niu.liu@nju.edu.cn)
"""

# -----------------------------  MAIN -----------------------------
from astropy.table import Table
import numpy as np
import time
# My modules
from vsh_fit_201124 import vsh_fit as vsh_fit_old
from vsh_fit import vsh_fit as vsh_fit_new
from pmt_convert_201124 import st_to_rotgldquad as st_to_rotgldquad_old
from pmt_convert import st_to_rotgldquad as st_to_rotgldquad_new
from generate_test_data import generate_test_sample

# Read the simulated data
rot_vec = np.array([20, 30, 15])
gli_vec = np.array([30, 24, 12])
qua_vec = np.array([-10, 20, -3, 5, 30, 9, 12, 39, 40, 12])
pmt_vec = np.concatenate((gli_vec, rot_vec, qua_vec))
test_tab = generate_test_sample(int(2e6), pmt_vec)

# Transform astropy.Column into np.array and mas -> uas
dra = np.array(test_tab["dra"])
ddec = np.array(test_tab["ddec"])
ra = np.array(test_tab["ra"])
dec = np.array(test_tab["dec"])
dra_err = np.array(test_tab["dra_err"])
ddec_err = np.array(test_tab["ddec_err"])
dra_ddc_cor = np.array(test_tab["dra_ddec_cor"])

print("#        "
      "               Glide [uas]             "
      "               Rotation [uas]    Time_taken [s]")
print("#        "
      "     G1            G2            G3        "
      "     R1            R2            R3        ")
print("#        "
      "-----------------------------------------   "
      "-----------------------------------------")

print("Input "
      "{0:4.0f}        0 {1:4.0f}        0 "
      "{2:4.0f}        0 {3:4.0f}        0 "
      "{4:4.0f}        0 {5:4.0f}        0 ".format(*pmt_vec[:6]))

# Old one
time_s = time.time()
pmt, err, cor_mat = vsh_fit_old(
    dra, ddec, dra_err, ddec_err, ra, dec, ra_dc_cor=dra_ddc_cor,
    l_max=2, pos_in_rad=True)
time_d = time.time()
time_delta = time_d - time_s

pmt1, err1, cor_mat1 = st_to_rotgldquad_old(pmt, err, cor_mat)

print("Old:  "
      "{0:4.0f}     {6:4.0f} {1:4.0f}     {7:4.0f} "
      "{2:4.0f}     {8:4.0f} {3:4.0f}     {9:4.0f} "
      "{4:4.0f}     {10:4.0f} {5:4.0f}     {11:4.0f}  "
      "{12:.0f}".format(*pmt1[:6], *err1[:6], time_delta))

# New one
time_s = time.time()
pmt, err, cor_mat = vsh_fit_new(
    dra, ddec, dra_err, ddec_err, ra, dec, l_max=2,
    ra_dc_cor=dra_ddc_cor, pos_in_rad=True)
time_d = time.time()
time_delta = time_d - time_s

pmt1, err1, cor_mat1 = st_to_rotgldquad_new(pmt, err, cor_mat)

print("New:  "
      "{0:4.0f}     {6:4.0f} {1:4.0f}     {7:4.0f} "
      "{2:4.0f}     {8:4.0f} {3:4.0f}     {9:4.0f} "
      "{4:4.0f}     {10:4.0f} {5:4.0f}     {11:4.0f}  "
      "{12:.0f}".format(*pmt1[:6], *err1[:6], time_delta))
# --------------------------------- END --------------------------------
