# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest

import vector

ak = pytest.importorskip("awkward")
numba = pytest.importorskip("numba")
pytest.importorskip("vector._backends.numba_object")


pytestmark = [pytest.mark.numba, pytest.mark.awkward]


def test():
    @numba.njit
    def extract(x):
        return x[2][0]

    array = vector.Array([[{"x": 1, "y": 2}], [], [{"x": 3, "y": 4}, {"x": 5, "y": 6}]])
    out = extract(array)
    assert isinstance(out, vector._backends.object_.VectorObject2D)
    assert out.x == pytest.approx(3)
    assert out.y == pytest.approx(4)

    array = vector.Array(
        [[{"x": 1, "y": 2, "z": 3, "E": 4}], [], [{"x": 5, "y": 6, "z": 7, "E": 15}]]
    )
    out = extract(array)
    assert isinstance(out, vector._backends.object_.MomentumObject4D)
    assert out.x == pytest.approx(5)
    assert out.y == pytest.approx(6)
    assert out.z == pytest.approx(7)
    assert out.t == pytest.approx(15)
