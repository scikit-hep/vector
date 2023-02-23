# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector.backends.object


def test_xy_z_t():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
        temporal=vector.backends.object.TemporalObjectT(20),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(3 / 20)
    assert out.y == pytest.approx(4 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_xy_z_tau():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
        temporal=vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(3 / 20)
    assert out.y == pytest.approx(4 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_xy_theta_t():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        temporal=vector.backends.object.TemporalObjectT(20),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(3 / 20)
    assert out.y == pytest.approx(4 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_xy_theta_tau():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        temporal=vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(3 / 20)
    assert out.y == pytest.approx(4 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_xy_eta_t():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        temporal=vector.backends.object.TemporalObjectT(20),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(3 / 20)
    assert out.y == pytest.approx(4 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_xy_eta_tau():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        temporal=vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(3 / 20)
    assert out.y == pytest.approx(4 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_rhophi_z_t():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
        temporal=vector.backends.object.TemporalObjectT(20),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(5 / 20)
    assert out.y == pytest.approx(0 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_rhophi_z_tau():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
        temporal=vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(5 / 20)
    assert out.y == pytest.approx(0 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_rhophi_theta_t():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        temporal=vector.backends.object.TemporalObjectT(20),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(5 / 20)
    assert out.y == pytest.approx(0 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_rhophi_theta_tau():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        temporal=vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(5 / 20)
    assert out.y == pytest.approx(0 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_rhophi_eta_t():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        temporal=vector.backends.object.TemporalObjectT(20),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(5 / 20)
    assert out.y == pytest.approx(0 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_rhophi_eta_tau():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        temporal=vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(5 / 20)
    assert out.y == pytest.approx(0 / 20)
    assert out.z == pytest.approx(10 / 20)
