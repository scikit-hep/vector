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


def test_necessary_columns():
    vec = vector.Array([[{"pt": 1, "phi": 2}], [], [{"pt": 3, "phi": 4}]])
    dak_vec = dak.from_awkward(vec, npartitions=1)

    cols = next(iter(dak.report_necessary_columns(dak_vec).values()))

    # this may seem weird at first: why would one need "phi" and "rho", if one asked for "pt"?
    # the reason is that vector will build internally a class with "phi" and "rho",
    # see: https://github.com/scikit-hep/vector/blob/608da2d55a74eed25635fd408d1075b568773c99/src/vector/backends/awkward.py#L166-L167
    # So, even if one asks for "pt", "phi" and "rho" are as well in order to build the vector class in the first place.
    # (the same argument holds true for all other vector classes)
    assert cols == frozenset({"phi", "rho"})
