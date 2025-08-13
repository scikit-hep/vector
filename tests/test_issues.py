# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
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

    file_path = os.path.join("tests", "samples", "issue-161-v2.pkl")

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
    azp2 = vector.backends.numpy.AzimuthalNumpyRhoPhi(
        [(2.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)],
        dtype=[("rho", float), ("phi", float)],
    )
    assert vec2d.azimuthal == az1
    assert vec2d.azimuthal != az2
    assert vec2d.azimuthal != azp1
    assert az1 != az2
    assert not az1 == azp1  # noqa: SIM201
    assert not azp1 == az1  # noqa: SIM201
    assert azp1 != az1
    assert azp1 == azp1  # noqa: PLR0124
    assert azp1 != azp2

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
    lgeta1 = vector.backends.numpy.LongitudinalNumpyEta(
        [(3.1,), (3.2,), (3.3,), (3.4,), (3.5,)], dtype=[("eta", float)]
    )
    lgeta2 = vector.backends.numpy.LongitudinalNumpyEta(
        [(4.1,), (3.2,), (3.3,), (3.4,), (3.5,)], dtype=[("eta", float)]
    )
    lgtheta1 = vector.backends.numpy.LongitudinalNumpyTheta(
        [(3.1,), (3.2,), (3.3,), (3.4,), (3.5,)], dtype=[("theta", float)]
    )
    lgtheta2 = vector.backends.numpy.LongitudinalNumpyTheta(
        [(4.1,), (3.2,), (3.3,), (3.4,), (3.5,)], dtype=[("theta", float)]
    )
    assert vec3d.azimuthal == az1
    assert vec3d.longitudinal == lg1
    assert vec3d.longitudinal != lg2
    assert vec3d.longitudinal != lgeta1
    assert lg1 != lg2
    assert not lg1 == lgeta1  # noqa: SIM201
    assert not lgeta1 == lg1  # noqa: SIM201
    assert lgeta1 != lg1
    assert lgeta1 == lgeta1  # noqa: PLR0124
    assert lgeta1 != lgeta2
    assert lgtheta1 == lgtheta1  # noqa: PLR0124
    assert lgtheta1 != lgtheta2
    assert lgtheta1 != lgeta1
    assert not lgtheta1 == lgeta1  # noqa: SIM201

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
    tmtau1 = vector.backends.numpy.TemporalNumpyTau(
        [(4.1,), (4.2,), (4.3,), (4.4,), (4.5,)], dtype=[("tau", float)]
    )
    tmtau2 = vector.backends.numpy.TemporalNumpyTau(
        [(5.1,), (4.2,), (4.3,), (4.4,), (4.5,)], dtype=[("tau", float)]
    )
    assert vec4d.azimuthal == az1
    assert vec4d.longitudinal == lg1
    assert vec4d.temporal == tm1
    assert vec4d.temporal != tm2
    assert vec4d.temporal != tmtau1
    assert tm1 != tm2
    assert not tm1 == tmtau1  # noqa: SIM201
    assert not tmtau1 == tm1  # noqa: SIM201
    assert tmtau1 != tm1
    assert tmtau1 == tmtau1  # noqa: PLR0124
    assert tmtau1 != tmtau2


def test_issue_463():
    v = vector.obj(x=1, y=1, z=1)
    for transform in "xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta":
        trv = getattr(v, "to_" + transform)()
        assert trv.deltaangle(trv) == 0.0


def test_issue_621():
    _ = pytest.importorskip("jax")
    ak = pytest.importorskip("awkward")
    vector.register_awkward()
    ak.jax.register_and_check()

    a = b = ak.to_backend(
        ak.zip({"x": [1], "y": [1], "z": [1], "t": [1]}, with_name="Momentum4D"), "jax"
    )

    # some computation that involves broadcast_and_apply in awkward
    # enough to check if it computes at all
    assert (a + b).mass
