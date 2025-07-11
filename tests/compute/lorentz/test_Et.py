# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import math

import pytest

import vector.backends.object


def test_xy_z_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectZ(10),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Et == pytest.approx(math.sqrt(80))


def test_xy_z_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectZ(10),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Et == pytest.approx(math.sqrt(80))


def test_xy_theta_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Et == pytest.approx(math.sqrt(80))


def test_xy_theta_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Et == pytest.approx(math.sqrt(80))


def test_xy_eta_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Et == pytest.approx(math.sqrt(80))


def test_xy_eta_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Et == pytest.approx(math.sqrt(80))


def test_rhophi_z_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectZ(10),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Et == pytest.approx(math.sqrt(80))


def test_rhophi_z_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectZ(10),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Et == pytest.approx(math.sqrt(80))


def test_rhophi_theta_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Et == pytest.approx(math.sqrt(80))


def test_rhophi_theta_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Et == pytest.approx(math.sqrt(80))


def test_rhophi_eta_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Et == pytest.approx(math.sqrt(80))


def test_rhophi_eta_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Et == pytest.approx(math.sqrt(80))
