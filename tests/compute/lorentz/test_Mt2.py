# Copyright (c) 2019, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector.backends.object


def test_xy_z_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectZ(10),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Mt2 == pytest.approx(300)


def test_xy_z_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectZ(10),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Mt2 == pytest.approx(300)


def test_xy_theta_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Mt2 == pytest.approx(300)


def test_xy_theta_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Mt2 == pytest.approx(300)


def test_xy_eta_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Mt2 == pytest.approx(300)


def test_xy_eta_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Mt2 == pytest.approx(300)


def test_rhophi_z_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectZ(10),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Mt2 == pytest.approx(300)


def test_rhophi_z_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectZ(10),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Mt2 == pytest.approx(300)


def test_rhophi_theta_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Mt2 == pytest.approx(300)


def test_rhophi_theta_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Mt2 == pytest.approx(300)


def test_rhophi_eta_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Mt2 == pytest.approx(300)


def test_rhophi_eta_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Mt2 == pytest.approx(300)


def test_transverse_spacelike_consistency():
    # Mt2 must be the signed t**2 - z**2 in both T- and Tau-coordinates
    # (no clamp to zero), so the two representations agree.
    base = vector.obj(px=3.0, py=4.0, pz=10.0, E=2.0)
    tau_vec = vector.obj(px=3.0, py=4.0, pz=10.0, tau=base.tau)
    assert base.Mt2 == pytest.approx(-96)
    assert tau_vec.Mt2 == pytest.approx(-96)
    assert base.Mt2 == pytest.approx(tau_vec.Mt2)
