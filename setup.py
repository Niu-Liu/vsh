#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
# File name: setup.py
"""
Created on Mon Feb  7 14:31:52 2022

@author: Neo(niu.liu@nju.edu.cn)
"""

from distutils.core import setup
# from distutils import find_packages

setup(
    name="vsh",
    version="0.0.1",
    #   py_modules=["vsh"],
    author="neo",
    author_email="niu.liu@foxmail.com",
    url="",
    description="Vector spherical harmonics",
    #   long_description_content_type="text/markdown",
    #   packages=find_packages(),
    packages=["vsh"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
