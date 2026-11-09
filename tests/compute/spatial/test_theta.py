# Copyright (c) 2019, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import math

import pytest

import vector
import vector.backends.object


def test_xy_z():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
    )
    assert vec.theta == pytest.approx(0.4636476090008061)


def test_xy_theta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
    )
    assert vec.theta == pytest.approx(0.4636476090008061)


def test_xy_eta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
    )
    assert vec.theta == pytest.approx(0.4636476090008061)


def test_rhophi_z():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
    )
    assert vec.theta == pytest.approx(0.4636476090008061)


def test_rhophi_theta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
    )
    assert vec.theta == pytest.approx(0.4636476090008061)


def test_rhophi_eta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
    )
    assert vec.theta == pytest.approx(0.4636476090008061)


@pytest.mark.parametrize("small", [1e-3, 1e-6, 1e-8, 1e-10, 1e-12])
def test_small_angle_precision(small):
    expected = math.atan(small)
    for vec in (
        vector.obj(x=small, y=0.0, z=1.0),
        vector.obj(x=0.0, y=-small, z=1.0),
        vector.obj(rho=small, phi=0.3, z=1.0),
        vector.obj(rho=-small, phi=0.3, z=1.0),
        vector.obj(px=small, py=0.0, pz=1.0, E=2.0),
    ):
        assert vec.theta == pytest.approx(expected, rel=1e-14, abs=0)

    arr = vector.array({"x": [small, 0.0], "y": [0.0, small], "z": [1.0, 1.0]})
    assert arr.theta == pytest.approx([expected, expected], rel=1e-14, abs=0)


def test_forward_momentum_precision():
    pt, pz = 0.01, 6800.0
    vec = vector.obj(pt=pt, phi=0.3, pz=pz, E=math.hypot(pt, pz))
    assert vec.theta == pytest.approx(math.atan2(pt, pz), rel=1e-14, abs=0)


def test_origin_and_poles():
    assert vector.obj(x=0.0, y=0.0, z=0.0).theta == 0.0
    assert vector.obj(x=0.0, y=0.0, z=2.0).theta == 0.0
    assert vector.obj(x=0.0, y=0.0, z=-2.0).theta == pytest.approx(math.pi)
    assert vector.obj(x=3.0, y=-4.0, z=0.0).theta == pytest.approx(math.pi / 2)
