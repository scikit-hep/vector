# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

dak = pytest.importorskip("dask_awkward")
ak = pytest.importorskip("awkward")


def test_constructor():
    x = dak.from_awkward(
        ak.Array([{"x": 1, "y": 2}, {"x": 1.1, "y": 2.2}]), npartitions=1
    )
    vec = vector.Array(x)

    assert isinstance(vec, dak.Array)
    assert isinstance(vec.compute(), vector.backends.awkward.VectorAwkward2D)
    assert ak.all(vec.x.compute() == ak.Array([1, 1.1]))
    assert ak.all(vec.y.compute() == ak.Array([2, 2.2]))
