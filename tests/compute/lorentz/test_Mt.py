# Copyright (c) 2019, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
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
    assert vec.Mt == pytest.approx(math.sqrt(300))


def test_xy_z_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectZ(10),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Mt == pytest.approx(math.sqrt(300))


def test_xy_theta_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Mt == pytest.approx(math.sqrt(300))


def test_xy_theta_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Mt == pytest.approx(math.sqrt(300))


def test_xy_eta_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Mt == pytest.approx(math.sqrt(300))


def test_xy_eta_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectXY(3, 4),
        vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Mt == pytest.approx(math.sqrt(300))


def test_rhophi_z_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectZ(10),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Mt == pytest.approx(math.sqrt(300))


def test_rhophi_z_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectZ(10),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Mt == pytest.approx(math.sqrt(300))


def test_rhophi_theta_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Mt == pytest.approx(math.sqrt(300))


def test_rhophi_theta_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Mt == pytest.approx(math.sqrt(300))


def test_rhophi_eta_t():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object.TemporalObjectT(20),
    )
    assert vec.Mt == pytest.approx(math.sqrt(300))


def test_rhophi_eta_tau():
    vec = vector.backends.object.MomentumObject4D(
        vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.Mt == pytest.approx(math.sqrt(300))


def test_transverse_spacelike_consistency():
    # For a transverse-spacelike vector (t**2 - z**2 < 0), ROOT's Mt() is
    # sign-preserving: Mt = copysign(sqrt(|Mt2|), Mt2). The T- and
    # Tau-coordinate representations of the same vector must agree.
    base = vector.obj(px=3.0, py=4.0, pz=10.0, E=2.0)
    expected = -math.sqrt(96)  # Mt2 = 4 - 100 = -96
    assert base.Mt2 == pytest.approx(-96)
    assert base.Mt == pytest.approx(expected)

    tau_vec = vector.obj(px=3.0, py=4.0, pz=10.0, tau=base.tau)
    assert tau_vec.Mt2 == pytest.approx(-96)
    assert tau_vec.Mt == pytest.approx(expected)
    assert base.Mt == pytest.approx(tau_vec.Mt)

    # all azimuthal/longitudinal representations agree (both T and Tau)
    for vec in (
        vector.obj(pt=base.pt, phi=base.phi, pz=10.0, E=2.0),
        vector.obj(px=3.0, py=4.0, theta=base.theta, E=2.0),
        vector.obj(px=3.0, py=4.0, eta=base.eta, E=2.0),
        vector.obj(pt=base.pt, phi=base.phi, theta=base.theta, tau=base.tau),
        vector.obj(px=3.0, py=4.0, eta=base.eta, tau=base.tau),
    ):
        assert vec.Mt2 == pytest.approx(-96)
        assert vec.Mt == pytest.approx(expected)


def test_timelike_consistency():
    # Sanity: ordinary timelike vector still matches between T and Tau coords.
    base = vector.obj(px=3.0, py=4.0, pz=0.0, E=10.0)
    tau_vec = vector.obj(px=3.0, py=4.0, pz=0.0, tau=base.tau)
    assert base.Mt == pytest.approx(10.0)
    assert tau_vec.Mt == pytest.approx(10.0)
