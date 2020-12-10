# -*- coding: utf-8 -*-
# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""


"""


from __future__ import absolute_import, division, print_function

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vector.protocols.lorentz import LorentzVector, Scalar

from vector.core import numpy as np


def p2(vec):
    # type: (LorentzVector) -> Scalar
    return vec.x ** 2 + vec.y ** 2 + vec.z ** 2


def p(vec):
    # type: (LorentzVector) -> Scalar
    return np.sqrt(p2(vec))


def beta(vec):
    # type: (LorentzVector) -> Scalar
    return p2(vec) / vec.t


def gamma(vec):
    # type: (LorentzVector) -> Scalar
    return 1 / np.sqrt(1 - beta(vec) ** 2)


def rapidity(vec):
    # type: (LorentzVector) -> Scalar
    return 0.5 * np.log((vec.t + vec.z) / (vec.t - vec.z))


def delta_r2(vec, other):
    # type: (LorentzVector, LorentzVector) -> Scalar
    """Return :math:`\\Delta R^2` the distance squared in (eta,phi) space with another Lorentz vector, defined as:
    :math:`\\Delta R^2 = (\\Delta \\eta)^2 + (\\Delta \\phi)^2`
    """
    delta_phi = np.mod(vec.phi - other.phi + np.pi, np.pi * 2) - np.pi
    return (vec.eta - other.eta) ** 2 + delta_phi ** 2


def delta_r(vec, other):
    # type: (LorentzVector, LorentzVector) -> Scalar
    """Return :math:`\\Delta R` the distance in (eta,phi) space with another Lorentz vector, defined as:
    :math:`\\Delta R = \\sqrt{(\\Delta \\eta)^2 + (\\Delta \\phi)^2}`
    """
    return np.sqrt(delta_r2(vec, other))
