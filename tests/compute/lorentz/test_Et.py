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


def test_negative_energy_sign_consistency():
    # ROOT's TLorentzVector::Et() is sign-preserving in the energy:
    # E < 0 ? -sqrt(Et2) : sqrt(Et2). All coordinate-system representations
    # of the same vector must therefore agree on the (negative) sign.
    xy_z = vector.obj(px=3.0, py=4.0, pz=0.0, E=-13.0)
    rhophi_z = vector.obj(pt=5.0, phi=0.0, pz=0.0, E=-13.0)
    assert xy_z.Et == pytest.approx(-13.0)
    assert rhophi_z.Et == pytest.approx(-13.0)
    assert xy_z.Et == pytest.approx(rhophi_z.Et)

    # non-axial pz, all four az/long combinations should still agree in sign
    px, py, pz, E = 3.0, 4.0, 10.0, -20.0
    expected = -math.sqrt(80)
    base = vector.obj(px=px, py=py, pz=pz, E=E)
    for vec in (
        base,
        vector.obj(pt=base.pt, phi=base.phi, pz=pz, E=E),
        vector.obj(px=px, py=py, theta=base.theta, E=E),
        vector.obj(px=px, py=py, eta=base.eta, E=E),
        vector.obj(pt=base.pt, phi=base.phi, theta=base.theta, E=E),
        vector.obj(pt=base.pt, phi=base.phi, eta=base.eta, E=E),
    ):
        assert vec.Et == pytest.approx(expected)
