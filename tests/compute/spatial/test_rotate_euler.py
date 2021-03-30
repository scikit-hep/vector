# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest

import vector._backends.object_


def test_spatial_object():
    vec = vector._backends.object_.VectorObject3D(
        vector._backends.object_.AzimuthalObjectXY(0.4, 0.5),
        vector._backends.object_.LongitudinalObjectZ(0.6),
    )
    out = vec.rotate_euler(0.1, 0.2, 0.3)
    assert isinstance(out, vector._backends.object_.VectorObject3D)
    assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
    assert isinstance(out.longitudinal, vector._backends.object_.LongitudinalObjectZ)
    assert out.x == pytest.approx(0.5956646364506655)
    assert out.y == pytest.approx(0.409927258162962)
    assert out.z == pytest.approx(0.4971350761081869)

    for t in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        tvec = getattr(vec, "to_" + t)()
        out = tvec.rotate_euler(0.1, 0.2, 0.3)
        assert isinstance(out, vector._backends.object_.VectorObject3D)
        assert isinstance(out.azimuthal, vector._backends.object_.AzimuthalObjectXY)
        assert isinstance(
            out.longitudinal, vector._backends.object_.LongitudinalObjectZ
        )
        assert out.x == pytest.approx(0.5956646364506655)
        assert out.y == pytest.approx(0.409927258162962)
        assert out.z == pytest.approx(0.4971350761081869)
