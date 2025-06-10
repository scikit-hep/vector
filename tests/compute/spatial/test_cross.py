# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import numpy
import pytest

import vector.backends.numpy
import vector.backends.object


def test_spatial_object():
    v1 = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.1, 0.2),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.3),
    )
    v2 = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.4, 0.5),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.6),
    )
    with pytest.raises(TypeError):
        out = v1.to_Vector4D().cross(v2)
    out = v1.cross(v2)
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert isinstance(out.azimuthal, vector.backends.object.AzimuthalObjectXY)
    assert isinstance(out.longitudinal, vector.backends.object.LongitudinalObjectZ)
    assert (out.x, out.y, out.z) == pytest.approx((-0.03, 0.06, -0.03))

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.cross(transformed2)
            assert isinstance(out, vector.backends.object.VectorObject3D)
            assert isinstance(out.azimuthal, vector.backends.object.AzimuthalObjectXY)
            assert isinstance(
                out.longitudinal, vector.backends.object.LongitudinalObjectZ
            )
            assert (out.x, out.y, out.z) == pytest.approx((-0.03, 0.06, -0.03))


def test_spatial_numpy():
    v1 = vector.backends.numpy.VectorNumpy3D(
        [(0.1, 0.2, 0.3)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    v2 = vector.backends.numpy.VectorNumpy3D(
        [(0.4, 0.5, 0.6)],
        dtype=[("x", numpy.float64), ("y", numpy.float64), ("z", numpy.float64)],
    )
    with pytest.raises(TypeError):
        out = v1.to_Vector4D().cross(v2)
    out = v1.cross(v2)
    assert isinstance(out, vector.backends.numpy.VectorNumpy3D)
    assert out.dtype.names == ("x", "y", "z")
    assert (out[0].x, out[0].y, out[0].z) == pytest.approx((-0.03, 0.06, -0.03))

    for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
            transformed1, transformed2 = (
                getattr(v1, "to_" + t1)(),
                getattr(v2, "to_" + t2)(),
            )
            out = transformed1.cross(transformed2)
            assert isinstance(out, vector.backends.numpy.VectorNumpy3D)
            assert out.dtype.names == ("x", "y", "z")
            assert (out[0].x, out[0].y, out[0].z) == pytest.approx((-0.03, 0.06, -0.03))
