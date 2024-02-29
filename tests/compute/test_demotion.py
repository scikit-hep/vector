# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import vector


def test_demotion_object():
    v1 = vector.obj(x=0.1, y=0.2)
    v2 = vector.obj(x=1, y=2, z=3)
    v3 = vector.obj(x=10, y=20, z=30, t=40)

    # order should not matter
    assert v1 + v2 == vector.obj(x=1.1, y=2.2)
    assert v2 + v1 == vector.obj(x=1.1, y=2.2)
    assert v1 + v3 == vector.obj(x=10.1, y=20.2)
    assert v3 + v1 == vector.obj(x=10.1, y=20.2)
    assert v2 + v3 == vector.obj(x=11, y=22, z=33)
    assert v3 + v2 == vector.obj(x=11, y=22, z=33)

    v1 = vector.obj(px=0.1, py=0.2)
    v2 = vector.obj(px=1, py=2, pz=3)
    v3 = vector.obj(px=10, py=20, pz=30, t=40)

    # order should not matter
    assert v1 + v2 == vector.obj(px=1.1, py=2.2)
    assert v2 + v1 == vector.obj(px=1.1, py=2.2)
    assert v1 + v3 == vector.obj(px=10.1, py=20.2)
    assert v3 + v1 == vector.obj(px=10.1, py=20.2)
    assert v2 + v3 == vector.obj(px=11, py=22, pz=33)
    assert v3 + v2 == vector.obj(px=11, py=22, pz=33)

    v1 = vector.obj(px=0.1, py=0.2)
    v2 = vector.obj(x=1, y=2, z=3)
    v3 = vector.obj(px=10, py=20, pz=30, t=40)

    # momentum + generic = generic
    assert v1 + v2 == vector.obj(x=1.1, y=2.2)
    assert v2 + v1 == vector.obj(x=1.1, y=2.2)
    assert v1 + v3 == vector.obj(px=10.1, py=20.2)
    assert v3 + v1 == vector.obj(px=10.1, py=20.2)
    assert v2 + v3 == vector.obj(x=11, y=22, z=33)
    assert v3 + v2 == vector.obj(x=11, y=22, z=33)


def test_demotion_numpy():
    v1 = vector.array(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
        },
    )
    v2 = vector.array(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
            "z": [5.0, 1.0, 1.0],
        },
    )
    v3 = vector.array(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
            "z": [5.0, 1.0, 1.0],
            "t": [16.0, 31.0, 46.0],
        },
    )

    v1_v2 = vector.array(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
        },
    )
    v2_v3 = vector.array(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
            "z": [10.0, 2.0, 2.0],
        },
    )
    v1_v3 = vector.array(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
        },
    )

    # order should not matter
    assert all(v1 + v2 == v1_v2)
    assert all(v2 + v1 == v1_v2)
    assert all(v1 + v3 == v1_v3)
    assert all(v3 + v1 == v1_v3)
    assert all(v2 + v3 == v2_v3)
    assert all(v3 + v2 == v2_v3)

    v1 = vector.array(
        {
            "px": [10.0, 20.0, 30.0],
            "py": [-10.0, 20.0, 30.0],
        },
    )
    v2 = vector.array(
        {
            "px": [10.0, 20.0, 30.0],
            "py": [-10.0, 20.0, 30.0],
            "pz": [5.0, 1.0, 1.0],
        },
    )
    v3 = vector.array(
        {
            "px": [10.0, 20.0, 30.0],
            "py": [-10.0, 20.0, 30.0],
            "pz": [5.0, 1.0, 1.0],
            "t": [16.0, 31.0, 46.0],
        },
    )

    p_v1_v2 = vector.array(
        {
            "px": [20.0, 40.0, 60.0],
            "py": [-20.0, 40.0, 60.0],
        },
    )
    p_v2_v3 = vector.array(
        {
            "px": [20.0, 40.0, 60.0],
            "py": [-20.0, 40.0, 60.0],
            "pz": [10.0, 2.0, 2.0],
        },
    )
    p_v1_v3 = vector.array(
        {
            "px": [20.0, 40.0, 60.0],
            "py": [-20.0, 40.0, 60.0],
        },
    )

    # order should not matter
    assert all(v1 + v2 == p_v1_v2)
    assert all(v2 + v1 == p_v1_v2)
    assert all(v1 + v3 == p_v1_v3)
    assert all(v3 + v1 == p_v1_v3)
    assert all(v2 + v3 == p_v2_v3)
    assert all(v3 + v2 == p_v2_v3)

    v2 = vector.array(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
            "z": [5.0, 1.0, 1.0],
        },
    )

    # momentum + generic = generic
    assert all(v1 + v2 == v1_v2)
    assert all(v2 + v1 == v1_v2)
    assert all(v1 + v3 == v1_v3)
    assert all(v3 + v1 == v1_v3)
    assert all(v2 + v3 == v2_v3)
    assert all(v3 + v2 == v2_v3)
