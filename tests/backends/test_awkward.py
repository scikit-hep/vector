# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest

import vector

ak = pytest.importorskip("awkward")


def test_basic():
    array = vector.Array([[{"x": 1, "y": 2}], [], [{"x": 3, "y": 4}]])
    assert isinstance(array, vector.backends.awkward_.VectorArray2D)
    assert array.x.tolist() == [[1], [], [3]]
    assert array.y.tolist() == [[2], [], [4]]
    assert array.rho.tolist() == [[2.23606797749979], [], [5]]
    assert array.phi.tolist() == [[1.1071487177940904], [], [0.9272952180016122]]
    assert isinstance(array[2, 0], vector.backends.awkward_.VectorRecord2D)
    assert array[2, 0].rho == 5
    assert array.deltaphi(array).tolist() == [[0], [], [0]]

    array = vector.Array([[{"pt": 1, "phi": 2}], [], [{"pt": 3, "phi": 4}]])
    assert isinstance(array, vector.backends.awkward_.MomentumArray2D)

    array = vector.Array(
        [
            [{"x": 1, "y": 2, "z": 3, "wow": 99}],
            [],
            [{"x": 4, "y": 5, "z": 6, "wow": 123}],
        ]
    )
    assert isinstance(array, vector.backends.awkward_.VectorArray3D)
    assert array.wow.tolist() == [[99], [], [123]]
