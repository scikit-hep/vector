# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import vector
from vector import (
    MomentumNumpy2D,
    MomentumNumpy3D,
    MomentumNumpy4D,
    MomentumObject2D,
    MomentumObject3D,
    MomentumObject4D,
    VectorObject4D,
)


def test_handler_of():
    object_a = VectorObject4D.from_xyzt(0.0, 0.0, 0.0, 0.0)
    object_b = VectorObject4D.from_xyzt(1.0, 1.0, 1.0, 1.0)
    protocol = vector._methods._handler_of(object_a, object_b)
    assert protocol == object_a


def test_momentum_coordinate_transforms():
    numpy_vec = vector.array(
        {
            "px": [1.0, 2.0, 3.0],
            "py": [-1.0, 2.0, 3.0],
        },
    )
    object_vec = MomentumObject2D(px=0.0, py=0.0)

    for t1 in "pxpy", "ptphi":
        for t2 in "pz", "eta", "theta":
            for t3 in "mass", "energy":
                transformed_object = getattr(object_vec, "to_" + t1)()
                assert isinstance(transformed_object, MomentumObject2D)
                assert hasattr(transformed_object, t1[:2])
                assert hasattr(transformed_object, t1[2:])

                transformed_object = getattr(object_vec, "to_" + t1 + t2)()
                assert isinstance(transformed_object, MomentumObject3D)
                assert hasattr(transformed_object, t1[:2])
                assert hasattr(transformed_object, t1[2:])
                assert hasattr(transformed_object, t2)

                transformed_object = getattr(object_vec, "to_" + t1 + t2 + t3)()
                assert isinstance(transformed_object, MomentumObject4D)
                assert hasattr(transformed_object, t1[:2])
                assert hasattr(transformed_object, t1[2:])
                assert hasattr(transformed_object, t2)
                assert hasattr(transformed_object, t3)

                transformed_numpy = getattr(numpy_vec, "to_" + t1)()
                assert isinstance(transformed_numpy, MomentumNumpy2D)
                assert hasattr(transformed_numpy, t1[:2])
                assert hasattr(transformed_numpy, t1[2:])

                transformed_numpy = getattr(numpy_vec, "to_" + t1 + t2)()
                assert isinstance(transformed_numpy, MomentumNumpy3D)
                assert hasattr(transformed_numpy, t1[:2])
                assert hasattr(transformed_numpy, t1[2:])
                assert hasattr(transformed_numpy, t2)

                transformed_numpy = getattr(numpy_vec, "to_" + t1 + t2 + t3)()
                assert isinstance(transformed_numpy, MomentumNumpy4D)
                assert hasattr(transformed_numpy, t1[:2])
                assert hasattr(transformed_numpy, t1[2:])
                assert hasattr(transformed_numpy, t2)
                assert hasattr(transformed_numpy, t3)
