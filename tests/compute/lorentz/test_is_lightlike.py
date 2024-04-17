# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import vector.backends.object


def test_xy_z_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectT(1),
    )
    assert vec.is_lightlike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectT(2),
    )
    assert not vec.is_lightlike()


def test_xy_z_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectTau(0),
    )
    assert vec.is_lightlike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectTau(1.7320508075688772),
    )
    assert not vec.is_lightlike()


def test_xy_theta_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectT(1),
    )
    assert vec.is_lightlike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectT(2),
    )
    assert not vec.is_lightlike()


def test_xy_theta_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectTau(0),
    )
    assert vec.is_lightlike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectTau(1.7320508075688772),
    )
    assert not vec.is_lightlike()


def test_xy_eta_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectT(1),
    )
    assert vec.is_lightlike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectT(2),
    )
    assert not vec.is_lightlike()


def test_xy_eta_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectTau(0),
    )
    assert vec.is_lightlike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectTau(1.7320508075688772),
    )
    assert not vec.is_lightlike()


def test_rhophi_z_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectT(1),
    )
    assert vec.is_lightlike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectT(2),
    )
    assert not vec.is_lightlike()


def test_rhophi_z_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectTau(0),
    )
    assert vec.is_lightlike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectTau(-4.58257569495584),
    )
    assert not vec.is_lightlike()


def test_rhophi_theta_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectT(1),
    )
    assert vec.is_lightlike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectT(2),
    )
    assert not vec.is_lightlike()


def test_rhophi_theta_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectTau(0),
    )
    assert vec.is_lightlike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectTau(-4.58257569495584),
    )
    assert not vec.is_lightlike()


def test_rhophi_eta_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectT(1),
    )
    assert vec.is_lightlike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectT(2),
    )
    assert not vec.is_lightlike()


def test_rhophi_eta_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectTau(0),
    )
    assert vec.is_lightlike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectTau(-4.58257569495584),
    )
    assert not vec.is_lightlike()
