# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

import vector


def test_array_casting():
    obj = vector.obj(x=1, y=1)
    assert isinstance(obj, vector.VectorObject2D)
    assert isinstance(numpy.asanyarray(obj), vector.VectorNumpy2D)
    assert numpy.asanyarray(obj).shape == ()

    obj = vector.obj(px=1, py=1)
    assert isinstance(obj, vector.MomentumObject2D)
    assert isinstance(numpy.asanyarray(obj), vector.MomentumNumpy2D)
    assert numpy.asanyarray(obj).shape == ()

    obj = vector.obj(x=1, y=1, z=1)
    assert isinstance(obj, vector.VectorObject3D)
    assert isinstance(numpy.asanyarray(obj), vector.VectorNumpy3D)
    assert numpy.asanyarray(obj).shape == ()

    obj = vector.obj(px=1, py=1, pz=1)
    assert isinstance(obj, vector.MomentumObject3D)
    assert isinstance(numpy.asanyarray(obj), vector.MomentumNumpy3D)
    assert numpy.asanyarray(obj).shape == ()

    obj = vector.obj(x=1, y=1, z=1, t=1)
    assert isinstance(obj, vector.VectorObject4D)
    assert isinstance(numpy.asanyarray(obj), vector.VectorNumpy4D)
    assert numpy.asanyarray(obj).shape == ()

    obj = vector.obj(px=1, py=1, pz=1, E=1)
    assert isinstance(obj, vector.MomentumObject4D)
    assert isinstance(numpy.asanyarray(obj), vector.MomentumNumpy4D)
    assert numpy.asanyarray(obj).shape == ()
