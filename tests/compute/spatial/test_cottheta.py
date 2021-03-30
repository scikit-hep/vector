# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import math

import pytest

import vector._backends.object_


def test_xy_z():
    vec = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(3, 4),
        vector._backends.object_.LongitudinalObjectZ(10),
    )
    assert vec.cottheta == pytest.approx(1 / math.tan(0.4636476090008061))


def test_xy_theta():
    vec = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(3, 4),
        vector._backends.object_.LongitudinalObjectTheta(0.4636476090008061),
    )
    assert vec.cottheta == pytest.approx(1 / math.tan(0.4636476090008061))


def test_xy_eta():
    vec = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(3, 4),
        vector._backends.object_.LongitudinalObjectEta(1.4436354751788103),
    )
    assert vec.cottheta == pytest.approx(1 / math.tan(0.4636476090008061))


def test_rhophi_z():
    vec = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector._backends.object_.LongitudinalObjectZ(10),
    )
    assert vec.cottheta == pytest.approx(1 / math.tan(0.4636476090008061))


def test_rhophi_theta():
    vec = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector._backends.object_.LongitudinalObjectTheta(0.4636476090008061),
    )
    assert vec.cottheta == pytest.approx(1 / math.tan(0.4636476090008061))


def test_rhophi_eta():
    vec = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector._backends.object_.LongitudinalObjectEta(1.4436354751788103),
    )
    assert vec.cottheta == pytest.approx(1 / math.tan(0.4636476090008061))
