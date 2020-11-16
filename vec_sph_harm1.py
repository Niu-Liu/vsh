#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: vec_sph_harm1.py
"""
Created on Mon Nov 16 19:58:37 2020

@author: Neo(niu.liu@nju.edu.cn)
"""

import numpy as np
from numpy import pi
from math import factorial
import sympy
import legenp from mpmath

# -----------------------------  MAIN -----------------------------


def sign_sph_harm(l, m, ra, de):
    """Spherical function defined in Eq.(9)

    """

    str_expr = "(-1)^m * sph_harm(m, l, ra, de)"
    expr = sympy.simplify(str_expr)
    func = sympy.lambdify([l, m, ra, de], expr, "scipy")

    return func

# def jjj


def st_harm(l, m):
    """Toroidal and Spheroidal functions
    """

    # Generate two variables
    ra = sympy.Symbol("ra")
    de = sympy.Symbol("de")

    fac = (2*l+1) / l / (l+1) / (4 * pi) * factorial(l-m) / factorial(l+m)
    fac = (-1)**m * np.sqrt(fac) * sympy.exp(complex(0, m*ra))

    # Expressions for two coefficient functions
    def Alm(x): return sysmpy.sqrt(1-x**2)
##   par_sph_ra = sympy.diff(sign_sph_harm, ra)
##   par_sph_de = sympy.diff(sign_sph_harm, de)
#    fac = 1.0 / sympy.sqrt(l * (l+1))
#    t_ra = par_sph_de * fac
#    t_de = -par_sph_ra / sympy.cos(de) * fac
#    s_ra = -par_sph_ra / sympy.cos(de) * fac
#    s_de = par_sph_de * fac
#
#    return t_ra, t_de, s_ra, s_de


def main(l0, m0, ra0, de0):
    """Some tests
    """

    l = sympy.Symbol("l")
    m = sympy.Symbol("m")
    ra = sympy.Symbol("ra")
    de = sympy.Symbol("de")

    st_harm(l, m, ra, de)

#    t_ra, t_de, s_ra, s_de = st_harm(l, m, ra, de)
#
#    print("T_ra:", t_ra(l0, m0, ra0, de0))
#    print("T_de:", t_ra(l0, m0, ra0, de0))
#    print("S_ra:", s_ra(l0, m0, ra0, de0))
#    print("S_de:", s_ra(l0, m0, ra0, de0))


l = 5
m = 2
ra = pi / 3
de = pi / 4
main(l, m, ra, de)

# --------------------------------- END --------------------------------
