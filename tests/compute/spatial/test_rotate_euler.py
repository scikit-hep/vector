# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy
import pytest

import vector.backends.numpy_
import vector.backends.object_
import vector.geometry


def test_spatial_object():
    vec = vector.backends.object_.SpatialVectorObject(
        vector.backends.object_.AzimuthalObjectXY(0.1, 0.2),
        vector.backends.object_.LongitudinalObjectZ(0.3),
    )
    # out = vec.rotate_euler()
    # assert isinstance(out, vector.backends.object_.SpatialVectorObject)
    # assert isinstance(out.azimuthal, vector.backends.object_.AzimuthalObjectXY)
    # assert isinstance(out.longitudinal, vector.backends.object_.LongitudinalObjectZ)
    # assert out.x == pytest.approx(0.37483425404335763)
    # assert out.y == pytest.approx(0.5383405688588193)
    # assert out.z == pytest.approx(0.5828282027463345)

    # for t1 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
    #     for t2 in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
    #         taxis, tvec = (
    #             getattr(axis, "to_" + t1)(),
    #             getattr(vec, "to_" + t2)(),
    #         )
    #         out = tvec.rotate_axis(taxis, 0.25)
    #         assert isinstance(out, vector.backends.object_.SpatialVectorObject)
    #         assert isinstance(out.azimuthal, vector.backends.object_.AzimuthalObjectXY)
    #         assert isinstance(
    #             out.longitudinal, vector.backends.object_.LongitudinalObjectZ
    #         )
    #         assert out.x == pytest.approx(0.37483425404335763)
    #         assert out.y == pytest.approx(0.5383405688588193)
    #         assert out.z == pytest.approx(0.5828282027463345)
