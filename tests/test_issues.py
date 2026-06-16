# Copyright (c) 2019, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import os
import pickle
import subprocess
import sys

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


def test_issue_657():
    ak = pytest.importorskip("awkward")
    vector.register_awkward()

    v1_lorentz = vector.Array(
        [
            {"x": 1.0, "y": 2.0, "z": 3.0, "t": 4.0},
            {"x": 5.0, "y": 6.0, "z": 7.0, "t": 8.0},
        ],
    )

    v2_lorentz = vector.Array(
        [
            {"x": 1.0 + 1e-9, "y": 2.0 - 1e-9, "z": 3.0 + 1e-9, "t": 4.0 - 1e-9},
            {"x": 5.0 - 1e-9, "y": 6.0 + 1e-9, "z": 7.0 - 1e-9, "t": 8.0 + 1e-9},
        ]
    )

    fields_lorentz = (
        "xyzt",
        "xythetat",
        "xyetat",
        "rhophizt",
        "rhophithetat",
        "rhophietat",
        "xyztau",
        "xythetatau",
        "xyetatau",
        "rhophiztau",
        "rhophithetatau",
        "rhophietatau",
    )

    for t1 in fields_lorentz:
        for t2 in fields_lorentz:
            transformed_v1, transformed_v2 = (
                getattr(v1_lorentz, "to_" + t1)(),
                getattr(v2_lorentz, "to_" + t2)(),
            )
            assert ak.all(transformed_v1.isclose(transformed_v2))
            assert not ak.all(
                transformed_v1.isclose(transformed_v2, rtol=1e-10, atol=1e-10)
            )

            # test ak.Record
            transformed_r1 = transformed_v1[0]
            transformed_r2 = transformed_v2[0]

            assert transformed_r1.isclose(transformed_r2)
            assert not transformed_r1.isclose(transformed_r2, rtol=1e-10, atol=1e-10)

    v1_3d = v1_lorentz.to_xyz()
    v2_3d = v2_lorentz.to_xyz()

    fields_3d = ("xyz", "xytheta", "xyeta", "rhophiz", "rhophitheta", "rhophieta")

    for t1 in fields_3d:
        for t2 in fields_3d:
            transformed_v1, transformed_v2 = (
                getattr(v1_3d, "to_" + t1)(),
                getattr(v2_3d, "to_" + t2)(),
            )
            assert ak.all(transformed_v1.isclose(transformed_v2))
            assert not ak.all(
                transformed_v1.isclose(transformed_v2, rtol=1e-10, atol=1e-10)
            )

            # test ak.Record
            transformed_r1 = transformed_v1[0]
            transformed_r2 = transformed_v2[0]

            assert transformed_r1.isclose(transformed_r2)
            assert not transformed_r1.isclose(transformed_r2, rtol=1e-10, atol=1e-10)

    v1_2d = v1_lorentz.to_xy()
    v2_2d = v2_lorentz.to_xy()

    fields_2d = ("xy", "rhophi")

    for t1 in fields_2d:
        for t2 in fields_2d:
            transformed_v1, transformed_v2 = (
                getattr(v1_2d, "to_" + t1)(),
                getattr(v2_2d, "to_" + t2)(),
            )
            assert ak.all(transformed_v1.isclose(transformed_v2))
            assert not ak.all(
                transformed_v1.isclose(transformed_v2, rtol=1e-10, atol=1e-10)
            )

            # test ak.Record
            transformed_r1 = transformed_v1[0]
            transformed_r2 = transformed_v2[0]

            assert transformed_r1.isclose(transformed_r2)
            assert not transformed_r1.isclose(transformed_r2, rtol=1e-10, atol=1e-10)


def test_issue_704():
    ak = pytest.importorskip("awkward")
    vector.register_awkward()

    vec_ak = ak.zip(
        {
            "pt": [1, 2, 3, 4],
            "eta": [1, 2, 3, 4],
            "phi": [1, 2, 3, 4],
            "mass": [1, 2, 3, 4],
        },
        with_name="Momentum4D",
    )
    vec_vec = vector.zip(
        {
            "pt": [1, 2, 3, 4],
            "eta": [1, 2, 3, 4],
            "phi": [1, 2, 3, 4],
            "mass": [1, 2, 3, 4],
        }
    )

    assert isinstance(vec_ak.neg3D, vector.backends.awkward.MomentumArray4D)
    assert isinstance(vec_vec.neg3D, vector.backends.awkward.MomentumArray4D)


def test_star_import_without_optional_deps():
    """from vector import * must not raise even when sympy/awkward are absent."""
    # Block sympy via a find_spec-based meta path finder inserted before vector is imported.
    # We also clear any cached sympy entries from sys.modules so the blocker takes effect.
    code = """
import sys

class _FailFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname == 'sympy' or fullname.startswith('sympy.'):
            from importlib.machinery import ModuleSpec
            return ModuleSpec(fullname, None)  # no loader -> ImportError
        return None

for k in list(sys.modules.keys()):
    if k == 'sympy' or k.startswith('sympy.'):
        del sys.modules[k]
sys.meta_path.insert(0, _FailFinder())

import vector
# from vector import * must not raise AttributeError
exec('from vector import *')
# All names in __all__ must be accessible (None for missing deps is acceptable)
for name in vector.__all__:
    _ = getattr(vector, name)
# __dir__ must not include sympy names when sympy is absent
d = dir(vector)
for name in ('VectorSympy', 'VectorSympy2D', 'VectorSympy3D', 'VectorSympy4D',
             'MomentumSympy2D', 'MomentumSympy3D', 'MomentumSympy4D'):
    assert name not in d, f"{name!r} should not appear in dir(vector) when sympy is absent"
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_like_non_vector_raises():
    """vector.like() must raise TypeError for non-vector arguments."""
    v = vector.obj(x=1.0, y=2.0, z=3.0)
    with pytest.raises(TypeError, match="is not a vector"):
        v.like(5)
    with pytest.raises(TypeError, match="is not a vector"):
        v.like("not a vector")


def test_to_vector4d_temporal_error_message():
    """Error message for over-specified temporal coords should say 'temporal', not 'longitudinal'."""
    v2 = vector.obj(x=1.0, y=2.0)
    with pytest.raises(TypeError, match="temporal"):
        v2.to_Vector4D(t=1.0, tau=2.0)
    v3 = vector.obj(x=1.0, y=2.0, z=3.0)
    with pytest.raises(TypeError, match="temporal"):
        v3.to_Vector4D(t=1.0, mass=2.0)


def test_dir_excludes_missing_backend_names(monkeypatch):
    """__dir__ drops awkward/sympy names when the backend module is unavailable."""
    monkeypatch.setattr(vector, "awkward", None)
    monkeypatch.setattr(vector, "sympy", None)
    names = set(dir(vector))
    assert not (names & vector._AWKWARD_NAMES)
    assert not (names & vector._SYMPY_NAMES)

    monkeypatch.setattr(vector, "awkward", object())
    monkeypatch.setattr(vector, "sympy", object())
    names = set(dir(vector))
    assert names >= vector._AWKWARD_NAMES
    assert names >= vector._SYMPY_NAMES
