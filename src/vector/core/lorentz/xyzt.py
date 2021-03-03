# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.


from vector.core import numpy as np
from vector.protocols.lorentz import LorentzTuple, LorentzVector, Scalar

# Functions that return a value

# x: already included
# y: already included
# z: already included
# t: already included


def add(left: LorentzVector, right: LorentzVector) -> LorentzTuple:
    x = left.x + right.x
    y = left.y + right.y
    z = left.z + right.z
    t = left.t + right.t

    return x, y, z, t


def add_scalar(left: LorentzVector, right: Scalar) -> LorentzTuple:
    x = left.x + right
    y = left.y + right
    z = left.z + right
    t = left.t + right

    return x, y, z, t


def dot(left: LorentzVector, right: LorentzVector) -> Scalar:
    return left.t * right.t - left.x * right.x - left.y * right.y - left.z * right.z


def multiply_scalar(left: LorentzVector, right: Scalar) -> LorentzTuple:
    return left.x * right, left.y * right, left.z * right, left.t * right


def pt(vec: LorentzVector) -> Scalar:
    return np.sqrt(vec.x ** 2 + vec.y ** 2)


def eta(vec: LorentzVector) -> Scalar:
    return np.arcsinh(vec.z / np.sqrt(vec.x ** 2 + vec.y ** 2))


def phi(vec: LorentzVector) -> Scalar:
    return np.arctan2(vec.y, vec.x)


# mass
def mag(vec: LorentzVector) -> Scalar:
    return np.sqrt(vec.t ** 2 - vec.x ** 2 - vec.y ** 2 - vec.z ** 2)


# mass2
def mag2(vec: LorentzVector) -> Scalar:
    return vec.t ** 2 - vec.x ** 2 - vec.y ** 2 - vec.z ** 2
