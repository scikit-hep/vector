# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import os

import pytest

import vector

ak = pytest.importorskip("awkward")
numba = pytest.importorskip("numba")
pytest.importorskip("vector.backends._numba_object")


pytestmark = [pytest.mark.numba, pytest.mark.awkward]


# awkward._v2 has not yet registered Numba dispatch mechanisms
# see https://github.com/scikit-hep/awkward/discussions/1639
# TODO: ensure this passes once awkward v2 is out
@pytest.mark.xfail(
    strict=True if os.environ.get("VECTOR_USE_AWKWARDV2") is not None else False
)
def test():
    @numba.njit
    def extract(x):
        return x[2][0]

    array = vector.Array([[{"x": 1, "y": 2}], [], [{"x": 3, "y": 4}, {"x": 5, "y": 6}]])
    out = extract(array)
    assert isinstance(out, vector.backends.object.VectorObject2D)
    assert out.x == pytest.approx(3)
    assert out.y == pytest.approx(4)

    array = vector.Array(
        [[{"x": 1, "y": 2, "z": 3, "E": 4}], [], [{"x": 5, "y": 6, "z": 7, "E": 15}]]
    )
    out = extract(array)
    assert isinstance(out, vector.backends.object.MomentumObject4D)
    assert out.x == pytest.approx(5)
    assert out.y == pytest.approx(6)
    assert out.z == pytest.approx(7)
    assert out.t == pytest.approx(15)
