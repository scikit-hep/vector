# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector.backends.object


def test_spatial_object():
    vec = vector.backends.object.VectorObject3D(
        azimuthal=vector.backends.object.AzimuthalObjectXY(0.4, 0.5),
        longitudinal=vector.backends.object.LongitudinalObjectZ(0.6),
    )
    out = vec.rotate_euler(0.1, 0.2, 0.3)
    assert isinstance(out, vector.backends.object.VectorObject3D)
    assert isinstance(out.azimuthal, vector.backends.object.AzimuthalObjectXY)
    assert isinstance(out.longitudinal, vector.backends.object.LongitudinalObjectZ)
    assert out.x == pytest.approx(0.5956646364506655)
    assert out.y == pytest.approx(0.409927258162962)
    assert out.z == pytest.approx(0.4971350761081869)

    for t in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        tvec = getattr(vec, "to_" + t)()
        out = tvec.rotate_euler(0.1, 0.2, 0.3)
        assert isinstance(out, vector.backends.object.VectorObject3D)
        assert isinstance(out.azimuthal, vector.backends.object.AzimuthalObjectXY)
        assert isinstance(out.longitudinal, vector.backends.object.LongitudinalObjectZ)
        assert out.x == pytest.approx(0.5956646364506655)
        assert out.y == pytest.approx(0.409927258162962)
        assert out.z == pytest.approx(0.4971350761081869)
