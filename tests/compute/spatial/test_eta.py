# Copyright (c) 2019, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector.backends.object


def test_xy_z():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
    )
    assert vec.eta == pytest.approx(1.4436354751788103)


def test_xy_theta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
    )
    assert vec.eta == pytest.approx(1.4436354751788103)


def test_xy_eta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
    )
    assert vec.eta == pytest.approx(1.4436354751788103)


def test_rhophi_z():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
    )
    assert vec.eta == pytest.approx(1.4436354751788103)


def test_rhophi_theta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
    )
    assert vec.eta == pytest.approx(1.4436354751788103)


def test_rhophi_eta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
    )
    assert vec.eta == pytest.approx(1.4436354751788103)


def test_theta_out_of_range_consistency():
    # For theta outside (0, pi), -log(tan(theta/2)) is nan; both the xy_theta
    # and rhophi_theta variants must apply the same nan->0 guard so the two
    # representations of the same direction agree.
    for theta in (-0.5, 2.0 * 3.141592653589793):
        xy = vector.backends.object.VectorObject3D(
            azimuthal=vector.backends.object.AzimuthalObjectXY(1.0, 1.0),
            longitudinal=vector.backends.object.LongitudinalObjectTheta(theta),
        )
        rhophi = vector.backends.object.VectorObject3D(
            azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(1.0, 0.0),
            longitudinal=vector.backends.object.LongitudinalObjectTheta(theta),
        )
        assert xy.eta == pytest.approx(0.0)
        assert rhophi.eta == pytest.approx(0.0)
        assert xy.eta == pytest.approx(rhophi.eta)
