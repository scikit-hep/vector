# -*- coding: utf-8 -*-
# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import absolute_import, division, print_function

from vector.core import numpy as np


def p2(vec):
    return vec.x ** 2 + vec.y ** 2 + vec.z ** 2


def p(vec):
    return np.sqrt(p2(vec))


def beta(vec):
    return p2(vec) / vec.t


def gamma(vec):
    return 1 / np.sqrt(1 - beta(vec) ** 2)


def rapidity(vec):
    return 0.5 * np.log((vec.t + vec.z) / (vec.t - vec.z))


def delta_r(vec, other):
    """Return :math:`\\Delta R` the distance in (eta,phi) space with another Lorentz vector, defined as:
    :math:`\\Delta R = \\sqrt{(\\Delta \\eta)^2 + (\\Delta \\phi)^2}`
    """
    delta_phi = np.mod(vec.phi - other.phi + np.pi, np.pi * 2) - np.pi
    return np.sqrt((vec.eta - other.eta) ** 2 + delta_phi ** 2)
