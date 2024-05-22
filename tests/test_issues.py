# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import os
import pickle

import numpy as np
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


def test_issue_443():
    ak = pytest.importorskip("awkward")
    vector.register_awkward()

    assert vector.array({"E": [1], "px": [1], "py": [1], "pz": [1]}) ** 2 == np.array(
        [-2.0]
    )
    assert ak.zip(
        {"E": [1], "px": [1], "py": [1], "pz": [1]}, with_name="Momentum4D"
    ) ** 2 == ak.Array([-2])
    assert vector.obj(E=1, px=1, py=1, pz=1) ** 2 == -2


def test_issue_194():
    vec2d = vector.VectorNumpy2D(
        {
            "x": [1.1, 1.2, 1.3, 1.4, 1.5],
            "y": [2.1, 2.2, 2.3, 2.4, 2.5],
        }
    )
    az1 = vector.backends.numpy.AzimuthalNumpyXY(
        [(1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)],
        dtype=[("x", float), ("y", float)],
    )
    az2 = vector.backends.numpy.AzimuthalNumpyXY(
        [(1.1, 3.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)],
        dtype=[("x", float), ("y", float)],
    )
    azp1 = vector.backends.numpy.AzimuthalNumpyRhoPhi(
        [(1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)],
        dtype=[("rho", float), ("phi", float)],
    )
    assert vec2d.azimuthal == az1
    assert vec2d.azimuthal != az2
    assert vec2d.azimuthal != azp1
    assert az1 != az2

    vec3d = vector.VectorNumpy3D(
        {
            "x": [1.1, 1.2, 1.3, 1.4, 1.5],
            "y": [2.1, 2.2, 2.3, 2.4, 2.5],
            "z": [3.1, 3.2, 3.3, 3.4, 3.5],
        }
    )
    lg1 = vector.backends.numpy.LongitudinalNumpyZ(
        [(3.1,), (3.2,), (3.3,), (3.4,), (3.5,)], dtype=[("z", float)]
    )
    lg2 = vector.backends.numpy.LongitudinalNumpyZ(
        [(4.1,), (3.2,), (3.3,), (3.4,), (3.5,)], dtype=[("z", float)]
    )
    lgeta = vector.backends.numpy.LongitudinalNumpyEta(
        [(3.1,), (3.2,), (3.3,), (3.4,), (3.5,)], dtype=[("eta", float)]
    )
    assert vec3d.azimuthal == az1
    assert vec3d.longitudinal == lg1
    assert vec3d.longitudinal != lg2
    assert vec3d.longitudinal != lgeta
    assert lg1 != lg2

    vec4d = vector.VectorNumpy4D(
        {
            "x": [1.1, 1.2, 1.3, 1.4, 1.5],
            "y": [2.1, 2.2, 2.3, 2.4, 2.5],
            "z": [3.1, 3.2, 3.3, 3.4, 3.5],
            "t": [4.1, 4.2, 4.3, 4.4, 4.5],
        }
    )
    tm1 = vector.backends.numpy.TemporalNumpyT(
        [(4.1,), (4.2,), (4.3,), (4.4,), (4.5,)], dtype=[("t", float)]
    )
    tm2 = vector.backends.numpy.TemporalNumpyT(
        [(5.1,), (4.2,), (4.3,), (4.4,), (4.5,)], dtype=[("t", float)]
    )
    tmtau = vector.backends.numpy.TemporalNumpyTau(
        [(4.1,), (4.2,), (4.3,), (4.4,), (4.5,)], dtype=[("tau", float)]
    )
    assert vec4d.azimuthal == az1
    assert vec4d.longitudinal == lg1
    assert vec4d.temporal == tm1
    assert vec4d.temporal != tm2
    assert vec4d.temporal != tmtau
    assert tm1 != tm2
