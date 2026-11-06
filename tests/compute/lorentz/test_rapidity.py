# Copyright (c) 2019, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import numpy
import pytest

import vector.backends.object


def test_xy_z_t():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
        temporal=vector.backends.object.TemporalObjectT(20),
    )
    assert vec.rapidity == pytest.approx(0.5493061443340549)


def test_xy_z_tau():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
        temporal=vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.rapidity == pytest.approx(0.5493061443340549)


def test_xy_theta_t():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        temporal=vector.backends.object.TemporalObjectT(20),
    )
    assert vec.rapidity == pytest.approx(0.5493061443340549)


def test_xy_theta_tau():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        temporal=vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.rapidity == pytest.approx(0.5493061443340549)


def test_xy_eta_t():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        temporal=vector.backends.object.TemporalObjectT(20),
    )
    assert vec.rapidity == pytest.approx(0.5493061443340549)


def test_xy_eta_tau():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        temporal=vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.rapidity == pytest.approx(0.5493061443340549)


def test_rhophi_z_t():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
        temporal=vector.backends.object.TemporalObjectT(20),
    )
    assert vec.rapidity == pytest.approx(0.5493061443340549)


def test_rhophi_z_tau():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
        temporal=vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.rapidity == pytest.approx(0.5493061443340549)


def test_rhophi_theta_t():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        temporal=vector.backends.object.TemporalObjectT(20),
    )
    assert vec.rapidity == pytest.approx(0.5493061443340549)


def test_rhophi_theta_tau():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
        temporal=vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.rapidity == pytest.approx(0.5493061443340549)


def test_rhophi_eta_t():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        temporal=vector.backends.object.TemporalObjectT(20),
    )
    assert vec.rapidity == pytest.approx(0.5493061443340549)


def test_rhophi_eta_tau():
    vec = vector.backends.object.VectorObject4D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
        temporal=vector.backends.object.TemporalObjectTau(16.583123951777),
    )
    assert vec.rapidity == pytest.approx(0.5493061443340549)


@pytest.mark.parametrize("pz", [5, 5.0])
def test_massless_along_z(pz):
    # a massless particle along the z axis has infinite rapidity, in either
    # direction, instead of raising ZeroDivisionError for E == pz
    with numpy.errstate(divide="ignore"):
        assert vector.obj(px=0, py=0, pz=pz, E=5).rapidity == numpy.inf
        assert vector.obj(px=0, py=0, pz=-pz, E=5).rapidity == -numpy.inf
