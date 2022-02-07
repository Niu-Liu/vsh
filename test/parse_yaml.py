#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: parse_yaml.py
"""
Created on Tue Nov 24 13:01:45 2020

@author: Neo(niu.liu@nju.edu.cn)
"""

import yaml

# -----------------------------  MAIN -----------------------------
input_file = open("config.yml")
env_par = yaml.load(input_file, Loader=yaml.FullLoader)

print(env_par)
# --------------------------------- END --------------------------------
