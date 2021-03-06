# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest

import vector.backends.object_


def test_xy_z():
    vec = vector.backends.object_.SpatialVectorObject(
        vector.backends.object_.AzimuthalObjectXY(3, 4),
        vector.backends.object_.LongitudinalObjectZ(10),
    )
    assert vec.z == pytest.approx(10)
