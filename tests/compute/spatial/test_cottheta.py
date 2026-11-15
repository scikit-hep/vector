# Copyright (c) 2019, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import math

import numpy
import pytest

import vector.backends.object


def test_xy_z():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
    )
    assert vec.cottheta == pytest.approx(1 / math.tan(0.4636476090008061))


def test_xy_theta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
    )
    assert vec.cottheta == pytest.approx(1 / math.tan(0.4636476090008061))


def test_xy_eta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(3, 4),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
    )
    assert vec.cottheta == pytest.approx(1 / math.tan(0.4636476090008061))


def test_rhophi_z():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectZ(10),
    )
    assert vec.cottheta == pytest.approx(1 / math.tan(0.4636476090008061))


def test_rhophi_theta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectTheta(0.4636476090008061),
    )
    assert vec.cottheta == pytest.approx(1 / math.tan(0.4636476090008061))


def test_rhophi_eta():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectRhoPhi(5, 0),
        longitudinal=vector.backends.object.LongitudinalObjectEta(1.4436354751788103),
    )
    assert vec.cottheta == pytest.approx(1 / math.tan(0.4636476090008061))


@pytest.mark.parametrize("z", [-3, 0, 3, -3.0, 0.0, 3.0])
@pytest.mark.parametrize("zero", [0, 0.0])
def test_rhophi_z_on_z_axis(z, zero):
    # rho == 0 gives the same cottheta as x == y == 0 instead of raising
    # ZeroDivisionError for Python numbers
    xy = vector.obj(x=zero, y=zero, z=z)
    rhophi = vector.obj(rho=zero, phi=zero, z=z)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        assert rhophi.cottheta == xy.cottheta
