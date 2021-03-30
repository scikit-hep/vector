# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest

import vector._backends.object_


def test_xy_z_t():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(3, 4),
        vector._backends.object_.LongitudinalObjectZ(10),
        vector._backends.object_.TemporalObjectT(20),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(3 / 20)
    assert out.y == pytest.approx(4 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_xy_z_tau():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(3, 4),
        vector._backends.object_.LongitudinalObjectZ(10),
        vector._backends.object_.TemporalObjectTau(16.583123951777),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(3 / 20)
    assert out.y == pytest.approx(4 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_xy_theta_t():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(3, 4),
        vector._backends.object_.LongitudinalObjectTheta(0.4636476090008061),
        vector._backends.object_.TemporalObjectT(20),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(3 / 20)
    assert out.y == pytest.approx(4 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_xy_theta_tau():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(3, 4),
        vector._backends.object_.LongitudinalObjectTheta(0.4636476090008061),
        vector._backends.object_.TemporalObjectTau(16.583123951777),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(3 / 20)
    assert out.y == pytest.approx(4 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_xy_eta_t():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(3, 4),
        vector._backends.object_.LongitudinalObjectEta(1.4436354751788103),
        vector._backends.object_.TemporalObjectT(20),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(3 / 20)
    assert out.y == pytest.approx(4 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_xy_eta_tau():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectXY(3, 4),
        vector._backends.object_.LongitudinalObjectEta(1.4436354751788103),
        vector._backends.object_.TemporalObjectTau(16.583123951777),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(3 / 20)
    assert out.y == pytest.approx(4 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_rhophi_z_t():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector._backends.object_.LongitudinalObjectZ(10),
        vector._backends.object_.TemporalObjectT(20),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(5 / 20)
    assert out.y == pytest.approx(0 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_rhophi_z_tau():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector._backends.object_.LongitudinalObjectZ(10),
        vector._backends.object_.TemporalObjectTau(16.583123951777),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(5 / 20)
    assert out.y == pytest.approx(0 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_rhophi_theta_t():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector._backends.object_.LongitudinalObjectTheta(0.4636476090008061),
        vector._backends.object_.TemporalObjectT(20),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(5 / 20)
    assert out.y == pytest.approx(0 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_rhophi_theta_tau():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector._backends.object_.LongitudinalObjectTheta(0.4636476090008061),
        vector._backends.object_.TemporalObjectTau(16.583123951777),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(5 / 20)
    assert out.y == pytest.approx(0 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_rhophi_eta_t():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector._backends.object_.LongitudinalObjectEta(1.4436354751788103),
        vector._backends.object_.TemporalObjectT(20),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(5 / 20)
    assert out.y == pytest.approx(0 / 20)
    assert out.z == pytest.approx(10 / 20)


def test_rhophi_eta_tau():
    vec = vector._backends.object_.VectorObject4D(
        vector._backends.object_.AzimuthalObjectRhoPhi(5, 0),
        vector._backends.object_.LongitudinalObjectEta(1.4436354751788103),
        vector._backends.object_.TemporalObjectTau(16.583123951777),
    )
    out = vec.to_beta3()
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert type(vec.azimuthal) == type(out.azimuthal)  # noqa: E721
    assert type(vec.longitudinal) == type(out.longitudinal)  # noqa: E721
    assert out.x == pytest.approx(5 / 20)
    assert out.y == pytest.approx(0 / 20)
    assert out.z == pytest.approx(10 / 20)
