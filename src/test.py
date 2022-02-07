#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: tune_para.py
"""
Created on Tue Nov 24 13:13:45 2020

@author: Neo(niu.liu@nju.edu.cn)
"""

# -----------------------------  MAIN -----------------------------

from astropy.table import Table
import numpy as np
import time

# My modules
from my_progs.vsh.vsh_fit import vsh_fit_4_table
from my_progs.vsh.generate_test_data import generate_test_sample

# Read the simulated data
rot_vec = np.array([20, 30, 15])
gli_vec = np.array([30, 24, 12])
# qua_vec = np.array([-10, 20, -3, 5, 30, 9, 12, 39, 40, 12])
qua_vec = np.zeros(10)
# pmt_vec = np.concatenate((gli_vec, rot_vec, qua_vec))
pmt_vec = np.concatenate((gli_vec, rot_vec))
test_tab = generate_test_sample(int(2e3), pmt_vec)

# Record the start time
time_s = time.time()

# DO the LSQ fitting
output = vsh_fit_4_table(test_tab, l_max=1, pos_in_rad=True)

# Record the end time
time_d = time.time()
# Time difference
time_delta = time_d - time_s
print("Time taken {:.2f} s.".format(time_delta))
# --------------------------------- END --------------------------------
