# Copyright (c) 2019-2025, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""Smoke tests for the Awkward backend on a CUDA device.

Each operation is computed on both the CPU and the GPU copy of the same
vectors, and the results are required to match.
"""

from __future__ import annotations

import numpy
import pytest

import vector

ak = pytest.importorskip("awkward")
cupy = pytest.importorskip("cupy")

if cupy.cuda.runtime.getDeviceCount() == 0:
    pytest.skip("no CUDA device found", allow_module_level=True)

pytestmark = [pytest.mark.awkward, pytest.mark.cuda]


@pytest.fixture
def vectors():
    """A pair of Momentum4D arrays, on the CPU and on the GPU."""
    left = vector.Array(
        [
            [{"px": 1.0, "py": 2.0, "pz": 3.0, "E": 10.0}],
            [],
            [
                {"px": -1.5, "py": 0.5, "pz": -2.0, "E": 5.0},
                {"px": 4.0, "py": -1.0, "pz": 0.25, "E": 12.0},
            ],
        ]
    )
    right = vector.Array(
        [
            [{"px": 0.5, "py": -2.0, "pz": 1.0, "E": 4.0}],
            [],
            [
                {"px": 2.5, "py": 1.5, "pz": 3.0, "E": 8.0},
                {"px": -3.0, "py": 2.0, "pz": -1.25, "E": 9.0},
            ],
        ]
    )
    return (left, right), (
        ak.to_backend(left, "cuda"),
        ak.to_backend(right, "cuda"),
    )


def assert_same(cpu_result, cuda_result):
    if not isinstance(cpu_result, ak.Array):
        assert cpu_result == pytest.approx(float(cuda_result))
        return

    assert ak.backend(cuda_result) == "cuda"
    cuda_result = ak.to_backend(cuda_result, "cpu")
    assert cpu_result.fields == cuda_result.fields

    for field in cpu_result.fields or [None]:
        left = ak.to_numpy(ak.ravel(cpu_result if field is None else cpu_result[field]))
        right = ak.to_numpy(
            ak.ravel(cuda_result if field is None else cuda_result[field])
        )
        if left.dtype == bool:
            numpy.testing.assert_array_equal(left, right)
        else:
            numpy.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)


UNARY = {
    "pt": lambda v: v.pt,
    "rho2": lambda v: v.rho2,
    "phi": lambda v: v.phi,
    "eta": lambda v: v.eta,
    "theta": lambda v: v.theta,
    "mag": lambda v: v.mag,
    "mass": lambda v: v.mass,
    "mass2": lambda v: v.mass2,
    "energy": lambda v: v.energy,
    "beta": lambda v: v.beta,
    "gamma": lambda v: v.gamma,
    "rapidity": lambda v: v.rapidity,
    "unit": lambda v: v.unit(),
    "neg3D": lambda v: -v,
    "to_rhophietatau": lambda v: v.to_rhophietatau(),
    "to_Vector2D": lambda v: v.to_Vector2D(),
    "to_Vector3D": lambda v: v.to_Vector3D(),
    "scale": lambda v: v * 2.5,
    "sum": lambda v: ak.sum(v.pt),
}

BINARY = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "dot": lambda a, b: a.dot(b),
    "deltaphi": lambda a, b: a.deltaphi(b),
    "deltaeta": lambda a, b: a.deltaeta(b),
    "deltaR": lambda a, b: a.deltaR(b),
    "deltaR2": lambda a, b: a.deltaR2(b),
    "deltaangle": lambda a, b: a.deltaangle(b),
    "cross": lambda a, b: a.to_Vector3D().cross(b.to_Vector3D()),
    "boost": lambda a, b: a.boost(b.to_beta3()),
    "boostCM_of": lambda a, b: a.boostCM_of(b),
    "is_parallel": lambda a, b: a.is_parallel(b),
    "equal": lambda a, b: a.equal(b),
}


@pytest.mark.parametrize("name", list(UNARY))
def test_unary(vectors, name):
    (cpu, _), (cuda, _) = vectors
    assert_same(UNARY[name](cpu), UNARY[name](cuda))


@pytest.mark.parametrize("name", list(BINARY))
def test_binary(vectors, name):
    (cpu_a, cpu_b), (cuda_a, cuda_b) = vectors
    assert_same(BINARY[name](cpu_a, cpu_b), BINARY[name](cuda_a, cuda_b))


def test_coordinate_systems(vectors):
    """Every 4D coordinate system round-trips back to px/py/pz/E on the GPU."""
    (cpu, _), (cuda, _) = vectors
    for azimuthal in ("xy", "rhophi"):
        for longitudinal in ("z", "theta", "eta"):
            for temporal in ("t", "tau"):
                convert = f"to_{azimuthal}{longitudinal}{temporal}"
                assert_same(
                    getattr(cpu, convert)().to_xyzt(),
                    getattr(cuda, convert)().to_xyzt(),
                )


def test_backend_is_preserved(vectors):
    _, (cuda_a, cuda_b) = vectors
    assert ak.backend(cuda_a + cuda_b) == "cuda"
    assert isinstance((cuda_a + cuda_b).px, ak.Array)
    assert cuda_a.pt.layout.backend.name == "cuda"
