# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import os
import pickle

import pytest

import vector


def test_issue_99():
    ak = pytest.importorskip("awkward")
    vector.register_awkward()
    vec = ak.Array([{"x": 1.0, "y": 2.0, "z": 3.0}], with_name="Vector3D")
    assert vec.to_xyz().tolist() == [{"x": 1.0, "y": 2.0, "z": 3.0}]
    assert vec[0].to_xyz().tolist() == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert vec[0].to_rhophiz().tolist() == {
        "rho": 2.23606797749979,
        "phi": 1.1071487177940904,
        "z": 3.0,
    }


def test_issue_161():
    ak = pytest.importorskip("awkward")
    nb = pytest.importorskip("numba")
    vector.register_awkward()

    @nb.njit
    def repro(generator_like_jet_constituents):
        for sublist in generator_like_jet_constituents:
            s = 0
            for generator_like_constituent in sublist:
                s += generator_like_constituent.pt

    file_path = (
        os.path.join("tests", "samples", "issue-161.pkl")
        if not vector._is_awkward_v2
        else os.path.join("tests", "samples", "issue-161-v2.pkl")
    )

    with open(file_path, "rb") as f:
        a = ak.from_buffers(*pickle.load(f))
    repro(generator_like_jet_constituents=a.constituents)
