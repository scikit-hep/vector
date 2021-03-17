# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest

import vector.backends.object_


def test_xy_z_t():
    vec = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectXY(3, 4),
        vector.backends.object_.LongitudinalObjectZ(10),
        vector.backends.object_.TemporalObjectT(20),
    )
    assert vec.gamma == pytest.approx(1.2060453783110545)


def test_xy_z_tau():
    vec = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectXY(3, 4),
        vector.backends.object_.LongitudinalObjectZ(10),
        vector.backends.object_.TemporalObjectTau(16.583123951777),
    )
    assert vec.gamma == pytest.approx(1.2060453783110545)


def test_xy_theta_t():
    vec = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectXY(3, 4),
        vector.backends.object_.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object_.TemporalObjectT(20),
    )
    assert vec.gamma == pytest.approx(1.2060453783110545)


def test_xy_theta_tau():
    vec = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectXY(3, 4),
        vector.backends.object_.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object_.TemporalObjectTau(16.583123951777),
    )
    assert vec.gamma == pytest.approx(1.2060453783110545)


def test_xy_eta_t():
    vec = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectXY(3, 4),
        vector.backends.object_.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object_.TemporalObjectT(20),
    )
    assert vec.gamma == pytest.approx(1.2060453783110545)


def test_xy_eta_tau():
    vec = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectXY(3, 4),
        vector.backends.object_.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object_.TemporalObjectTau(16.583123951777),
    )
    assert vec.gamma == pytest.approx(1.2060453783110545)


def test_rhophi_z_t():
    vec = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object_.LongitudinalObjectZ(10),
        vector.backends.object_.TemporalObjectT(20),
    )
    assert vec.gamma == pytest.approx(1.2060453783110545)


def test_rhophi_z_tau():
    vec = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object_.LongitudinalObjectZ(10),
        vector.backends.object_.TemporalObjectTau(16.583123951777),
    )
    assert vec.gamma == pytest.approx(1.2060453783110545)


def test_rhophi_theta_t():
    vec = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object_.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object_.TemporalObjectT(20),
    )
    assert vec.gamma == pytest.approx(1.2060453783110545)


def test_rhophi_theta_tau():
    vec = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object_.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object_.TemporalObjectTau(16.583123951777),
    )
    assert vec.gamma == pytest.approx(1.2060453783110545)


def test_rhophi_eta_t():
    vec = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object_.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object_.TemporalObjectT(20),
    )
    assert vec.gamma == pytest.approx(1.2060453783110545)


def test_rhophi_eta_tau():
    vec = vector.backends.object_.VectorObject4D(
        vector.backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object_.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object_.TemporalObjectTau(16.583123951777),
    )
    assert vec.gamma == pytest.approx(1.2060453783110545)
