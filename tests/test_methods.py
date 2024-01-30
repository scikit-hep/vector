# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector
from vector import VectorObject2D, VectorObject3D, VectorObject4D

awkward = pytest.importorskip("awkward")


def test_handler_of():
    object_a = VectorObject4D.from_xyzt(0.0, 0.0, 0.0, 0.0)
    object_b = VectorObject4D.from_xyzt(1.0, 1.0, 1.0, 1.0)
    protocol = vector._methods._handler_of(object_a, object_b)
    assert protocol == object_a

    object_a = VectorObject3D.from_xyz(0.0, 0.0, 0.0)
    object_b = VectorObject4D.from_xyzt(1.0, 1.0, 1.0, 1.0)
    protocol = vector._methods._handler_of(object_a, object_b)
    assert protocol == object_a

    object_a = VectorObject4D.from_xyzt(0.0, 0.0, 0.0, 0.0)
    object_b = VectorObject3D.from_xyz(1.0, 1.0, 1.0)
    protocol = vector._methods._handler_of(object_a, object_b)
    assert protocol == object_b

    object_a = VectorObject2D.from_xy(0.0, 0.0)
    object_b = VectorObject4D.from_xyzt(1.0, 1.0, 1.0, 1.0)
    protocol = vector._methods._handler_of(object_a, object_b)
    assert protocol == object_a

    object_a = VectorObject4D.from_xyzt(0.0, 0.0, 0.0, 0.0)
    object_b = VectorObject2D.from_xy(1.0, 1.0)
    protocol = vector._methods._handler_of(object_a, object_b)
    assert protocol == object_b

    object_a = VectorObject2D.from_xy(0.0, 0.0)
    object_b = VectorObject3D.from_xyz(1.0, 1.0, 1.0)
    protocol = vector._methods._handler_of(object_a, object_b)
    assert protocol == object_a

    object_a = VectorObject3D.from_xyz(0.0, 0.0, 0.0)
    object_b = VectorObject2D.from_xy(1.0, 1.0)
    protocol = vector._methods._handler_of(object_a, object_b)
    assert protocol == object_b

    awkward_a = vector.zip(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
            "z": [5.0, 10.0, 15.0],
            "t": [16.0, 31.0, 46.0],
        },
    )
    object_b = VectorObject2D.from_xy(1.0, 1.0)
    protocol = vector._methods._handler_of(awkward_a, object_b)
    # chooses awkward backend and converts the vector to 2D
    assert all(protocol == awkward_a.to_Vector2D())

    awkward_a = vector.zip(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
        },
    )
    object_b = VectorObject4D.from_xyzt(1.0, 1.0, 1.0, 1.0)
    protocol = vector._methods._handler_of(object_b, awkward_a)
    # chooses awkward backend and the vector is already of the
    # lower dimension
    assert all(protocol == awkward_a)

    awkward_a = vector.zip(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
        },
    )
    awkward_b = vector.zip(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [-1.0, 2.0, 3.0],
            "z": [5.0, 10.0, 15.0],
            "t": [16.0, 31.0, 46.0],
        },
    )
    object_b = VectorObject4D.from_xyzt(1.0, 1.0, 1.0, 1.0)
    protocol = vector._methods._handler_of(object_b, awkward_a, awkward_b)
    # chooses awkward backend and the 2D awkward vector
    # (first encountered awkward vector)
    assert all(protocol == awkward_a)

    awkward_a = vector.zip(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
            "z": [-10.0, 20.0, 30.0],
        },
    )
    awkward_b = vector.zip(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [-1.0, 2.0, 3.0],
            "z": [5.0, 10.0, 15.0],
            "t": [16.0, 31.0, 46.0],
        },
    )
    object_b = VectorObject2D.from_xy(1.0, 1.0)
    protocol = vector._methods._handler_of(awkward_b, object_b, awkward_a)
    # chooses awkward backend and converts awkward_b to 2D
    # (first encountered awkward vector)
    assert all(protocol == awkward_b.to_Vector2D())

    awkward_a = vector.zip(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
            "z": [5.0, 1.0, 1.0],
        },
    )
    awkward_b = vector.zip(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [-1.0, 2.0, 3.0],
            "z": [5.0, 10.0, 15.0],
            "t": [16.0, 31.0, 46.0],
        },
    )
    object_b = VectorObject2D.from_xy(1.0, 1.0)
    protocol = vector._methods._handler_of(object_b, awkward_a, awkward_b)
    # chooses awkward backend and converts the vector to 2D
    # (the first awkward vector encountered is used as the base)
    assert all(protocol == awkward_a.to_Vector2D())

    numpy_a = vector.array(
        {
            "x": [1.1, 1.2, 1.3, 1.4, 1.5],
            "y": [2.1, 2.2, 2.3, 2.4, 2.5],
            "z": [3.1, 3.2, 3.3, 3.4, 3.5],
        }
    )
    awkward_b = vector.zip(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [-1.0, 2.0, 3.0],
            "z": [5.0, 10.0, 15.0],
            "t": [16.0, 31.0, 46.0],
        },
    )
    object_b = VectorObject2D.from_xy(1.0, 1.0)
    protocol = vector._methods._handler_of(object_b, numpy_a, awkward_b)
    # chooses awkward backend and converts the vector to 2D
    assert all(protocol == awkward_b.to_Vector2D())

    awkward_a = vector.zip(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
            "z": [5.0, 1.0, 1.0],
        },
    )
    numpy_a = vector.array(
        {
            "x": [1.1, 1.2, 1.3, 1.4, 1.5],
            "y": [2.1, 2.2, 2.3, 2.4, 2.5],
        }
    )
    awkward_b = vector.zip(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [-1.0, 2.0, 3.0],
            "z": [5.0, 10.0, 15.0],
            "t": [16.0, 31.0, 46.0],
        },
    )
    object_b = VectorObject3D.from_xyz(1.0, 1.0, 1.0)
    protocol = vector._methods._handler_of(object_b, awkward_a, awkward_b, numpy_a)
    # chooses awkward backend and converts the vector to 2D
    # (the first awkward vector encountered is used as the base)
    assert all(protocol == awkward_a.to_Vector2D())
