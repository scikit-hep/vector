# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector.backends.object


def test_xy_z():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
    )
    assert vec.eta == pytest.approx(1.4436354751788103)


def test_xy_theta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
    )
    assert vec.eta == pytest.approx(1.4436354751788103)


def test_xy_eta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
    )
    assert vec.eta == pytest.approx(1.4436354751788103)


def test_rhophi_z():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
    )
    assert vec.eta == pytest.approx(1.4436354751788103)


def test_rhophi_theta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
    )
    assert vec.eta == pytest.approx(1.4436354751788103)


def test_rhophi_eta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
    )
    assert vec.eta == pytest.approx(1.4436354751788103)
