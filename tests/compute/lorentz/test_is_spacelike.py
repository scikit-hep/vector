# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import math

import vector.backends.object


def test_xy_z_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectZ(20),
        vector.backends.object.TemporalObjectT(math.sqrt(400)),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectZ(20),
        vector.backends.object.TemporalObjectT(math.sqrt(425)),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectZ(20),
        vector.backends.object.TemporalObjectT(math.sqrt(450)),
    )
    assert not vec.is_spacelike()


def test_xy_z_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectZ(20),
        vector.backends.object.TemporalObjectTau(-5),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectZ(20),
        vector.backends.object.TemporalObjectTau(2.384185791015625e-07),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectZ(20),
        vector.backends.object.TemporalObjectTau(5.000000000000005),
    )
    assert not vec.is_spacelike()


def test_xy_theta_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectTheta(0.24497866312686423),
        vector.backends.object.TemporalObjectT(math.sqrt(400)),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectTheta(0.24497866312686423),
        vector.backends.object.TemporalObjectT(math.sqrt(425)),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectTheta(0.24497866312686423),
        vector.backends.object.TemporalObjectT(math.sqrt(450)),
    )
    assert not vec.is_spacelike()


def test_xy_theta_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectTheta(0.24497866312686423),
        vector.backends.object.TemporalObjectTau(-5),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectTheta(0.24497866312686423),
        vector.backends.object.TemporalObjectTau(2.384185791015625e-07),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectTheta(0.24497866312686423),
        vector.backends.object.TemporalObjectTau(5.000000000000005),
    )
    assert not vec.is_spacelike()


def test_xy_eta_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectEta(2.0947125472611012),
        vector.backends.object.TemporalObjectT(math.sqrt(400)),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectEta(2.0947125472611012),
        vector.backends.object.TemporalObjectT(math.sqrt(425)),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectEta(2.0947125472611012),
        vector.backends.object.TemporalObjectT(math.sqrt(450)),
    )
    assert not vec.is_spacelike()


def test_xy_eta_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectEta(2.0947125472611012),
        vector.backends.object.TemporalObjectTau(-5),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectEta(2.0947125472611012),
        vector.backends.object.TemporalObjectTau(2.384185791015625e-07),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectEta(2.0947125472611012),
        vector.backends.object.TemporalObjectTau(5.000000000000005),
    )
    assert not vec.is_spacelike()


def test_rhophi_z_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectZ(20),
        vector.backends.object.TemporalObjectT(math.sqrt(400)),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectZ(20),
        vector.backends.object.TemporalObjectT(math.sqrt(425)),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectZ(20),
        vector.backends.object.TemporalObjectT(math.sqrt(450)),
    )
    assert not vec.is_spacelike()


def test_rhophi_z_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectZ(20),
        vector.backends.object.TemporalObjectTau(-5),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectZ(20),
        vector.backends.object.TemporalObjectTau(2.384185791015625e-07),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectZ(20),
        vector.backends.object.TemporalObjectTau(5.000000000000005),
    )
    assert not vec.is_spacelike()


def test_rhophi_theta_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectTheta(0.24497866312686423),
        vector.backends.object.TemporalObjectT(math.sqrt(400)),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectTheta(0.24497866312686423),
        vector.backends.object.TemporalObjectT(math.sqrt(425)),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectTheta(0.24497866312686423),
        vector.backends.object.TemporalObjectT(math.sqrt(450)),
    )
    assert not vec.is_spacelike()


def test_rhophi_theta_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectTheta(0.24497866312686423),
        vector.backends.object.TemporalObjectTau(-5),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectTheta(0.24497866312686423),
        vector.backends.object.TemporalObjectTau(2.384185791015625e-07),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectTheta(0.24497866312686423),
        vector.backends.object.TemporalObjectTau(5.000000000000005),
    )
    assert not vec.is_spacelike()


def test_rhophi_eta_t():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectEta(2.0947125472611012),
        vector.backends.object.TemporalObjectT(math.sqrt(400)),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectEta(2.0947125472611012),
        vector.backends.object.TemporalObjectT(math.sqrt(425)),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectEta(2.0947125472611012),
        vector.backends.object.TemporalObjectT(math.sqrt(450)),
    )
    assert not vec.is_spacelike()


def test_rhophi_eta_tau():
    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectEta(2.0947125472611012),
        vector.backends.object.TemporalObjectTau(-5),
    )
    assert vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectEta(2.0947125472611012),
        vector.backends.object.TemporalObjectTau(2.384185791015625e-07),
    )
    assert not vec.is_spacelike()

    vec = vector.backends.object.VectorObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectEta(2.0947125472611012),
        vector.backends.object.TemporalObjectTau(5.000000000000005),
    )
    assert not vec.is_spacelike()
