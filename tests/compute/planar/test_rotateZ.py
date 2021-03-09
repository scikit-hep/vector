# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

import vector.backends.numpy_
import vector.backends.object_


def test_xy():
    vec = vector.backends.object_.PlanarVectorObject(
        vector.backends.object_.AzimuthalObjectXY(1, 0)
    )
    print(vec.rotateZ(0.1))

    array = vector.backends.numpy_.PlanarVectorNumpy(
        [(0, 0), (1, 0), (0, 1)], dtype=[("x", numpy.float64), ("y", numpy.float64)]
    )
    assert isinstance(array.rotateZ(0.1), vector.backends.numpy_.PlanarVectorNumpy)
    assert array.rotateZ(0.1).dtype.names == ("x", "y")
    assert array.rotateZ(0.1).tolist() == [
        (0.0, 0.0),
        (0.9950041652780258, 0.09983341664682815),
        (-0.09983341664682815, 0.9950041652780258),
    ]


def test_rhophi():
    vec = vector.backends.object_.PlanarVectorObject(
        vector.backends.object_.AzimuthalObjectRhoPhi(1, 0)
    )
    print(vec.rotateZ(0.1))

    array = vector.backends.numpy_.PlanarVectorNumpy(
        [(0, 0), (1, 0), (0, 1)], dtype=[("rho", numpy.float64), ("phi", numpy.float64)]
    )
    assert isinstance(array.rotateZ(0.1), vector.backends.numpy_.PlanarVectorNumpy)
    assert array.rotateZ(0.1).dtype.names == ("rho", "phi")
    assert array.rotateZ(0.1).tolist() == [(0, 0.1), (1, 0.1), (0, 1.1)]
