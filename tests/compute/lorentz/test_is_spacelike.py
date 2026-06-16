# Copyright (c) 019-024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import vector.backends.object


def test_xy_z_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectT(0),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectT(1),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectT(2),
    )
    assert not vec.is_spacelike()


def test_xy_z_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectTau(-1),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectTau(0),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectTau(1.7320508075688772),
    )
    assert not vec.is_spacelike()


def test_xy_theta_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectT(0),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectT(1),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectT(2),
    )
    assert not vec.is_spacelike()


def test_xy_theta_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectTau(-1),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectTau(0),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectTau(1.7320508075688772),
    )
    assert not vec.is_spacelike()


def test_xy_eta_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectT(0),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectT(1),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectT(2),
    )
    assert not vec.is_spacelike()


def test_xy_eta_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectTau(-1),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectTau(0),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectTau(1.7320508075688772),
    )
    assert not vec.is_spacelike()


def test_rhophi_z_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectT(0),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectT(1),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectT(2),
    )
    assert not vec.is_spacelike()


def test_rhophi_z_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectTau(-1),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectTau(0),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectTau(1.7320508075688772),
    )
    assert not vec.is_spacelike()


def test_rhophi_theta_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectT(0),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectT(1),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectT(2),
    )
    assert not vec.is_spacelike()


def test_rhophi_theta_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectTau(-1),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectTau(0),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectTheta(1.5707963267948966),
        vector.backends.object.TemporalObjectTau(1.7320508075688772),
    )
    assert not vec.is_spacelike()


def test_rhophi_eta_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectT(0),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectT(1),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectT(2),
    )
    assert not vec.is_spacelike()


def test_rhophi_eta_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectTau(-1),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectTau(0),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(1, 0),
        vector.backends.object.LongitudinalObjectEta(0),
        vector.backends.object.TemporalObjectTau(1.7320508075688772),
    )
    assert not vec.is_spacelike()


def test_tolerance_partition():
    # Regression: with a nonzero tolerance, is_spacelike/is_lightlike/is_timelike
    # must partition the light cone without overlap. A nearly-lightlike timelike
    # vector should be lightlike only, never spacelike.
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(1, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectT(1.0000001),
    )
    assert not vec.is_spacelike(tolerance=1e-3)
    assert vec.is_lightlike(tolerance=1e-3)
    assert not vec.is_timelike(tolerance=1e-3)

    # A clearly timelike vector is not spacelike even with a nonzero tolerance.
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(0, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectT(2),
    )
    assert not vec.is_spacelike(tolerance=1e-3)
    assert vec.is_timelike(tolerance=1e-3)

    # A clearly spacelike vector remains spacelike with a nonzero tolerance.
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(2, 0),
        vector.backends.object.LongitudinalObjectZ(0),
        vector.backends.object.TemporalObjectT(0),
    )
    assert vec.is_spacelike(tolerance=1e-3)
