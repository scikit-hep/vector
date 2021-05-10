# This test code was written by the `hypothesis.extra.ghostwriter` module
# and is provided under the Creative Commons Zero public domain dedication.

import vector
from hypothesis import given, strategies as st

import pytest

# If ROOT is not available, skip these tests.
ROOT = pytest.importorskip("ROOT")

# Coordinate conversion methods to apply to the VectorObject4D.
coordinate_list = [
    "to_xyzt",
    "to_xythetat",
    "to_xyetat",
    "to_rhophizt",
    "to_rhophithetat",
    "to_rhophietat",
    "to_xyztau",
    "to_xythetatau",
    "to_xyetatau",
    "to_rhophiztau",
    "to_rhophithetatau",
    "to_rhophietatau",
]

@pytest.fixture(scope="module", params=coordinate_list)
def coordinates(request):
    return request.param

constructor = [
    (0, 0, 0, 0),
    (0, 0, 0, 10),
    (0, 0, 0, -10),
    (1, 2, 3, 0),
    (1, 2, 3, 10),
    (1, 2, 3, -10),
    (1, 2, 3, 2.5),
    (1, 2, 3, -2.5),
]

@pytest.mark.parametrize("constructor", constructor)
def test_M2(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).M2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().tau2
    )

@given(constructor=st.tuples(st.floats(), st.floats(), st.floats(), st.floats()) | st.tuples(st.integers(), st.integers(), st.integers(), st.integers()))
def test_fuzz_M2(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).M2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().tau2
    )

@given(azimuthal=st.tuples(st.floats(), st.floats()) | st.tuples(st.integers(), st.integers()))
def test_fuzz_MomentumObject2D(azimuthal):
    vec = vector.MomentumObject2D(azimuthal=azimuthal)


@given(azimuthal=st.tuples(st.floats(), st.floats()) | st.tuples(st.integers(), st.integers()), longitudinal=st.floats() | st.integers())
def test_fuzz_MomentumObject3D(azimuthal, longitudinal):
    vec = vector.MomentumObject3D(azimuthal=azimuthal, longitudinal=longitudinal)


@given(azimuthal=st.tuples(st.floats(), st.floats()) | st.tuples(st.integers(), st.integers()), longitudinal=st.floats() | st.integers(), temporal=st.floats() | st.integers())
def test_fuzz_MomentumObject4D(azimuthal, longitudinal, temporal):
    vec = vector.MomentumObject4D(
        azimuthal=azimuthal, longitudinal=longitudinal, temporal=temporal
    )


@given(azimuthal=st.tuples(st.floats(), st.floats()) | st.tuples(st.integers(), st.integers()))
def test_fuzz_VectorObject2D(azimuthal):
    vector.VectorObject2D(azimuthal=azimuthal)


@given(azimuthal=st.tuples(st.floats(), st.floats()) | st.tuples(st.integers(), st.integers()), longitudinal=st.floats() | st.integers())
def test_fuzz_VectorObject3D(azimuthal, longitudinal):
    vec = vector.VectorObject3D(azimuthal=azimuthal, longitudinal=longitudinal)


@given(azimuthal=st.tuples(st.floats(), st.floats()) | st.tuples(st.integers(), st.integers()), longitudinal=st.floats() | st.integers(), temporal=st.floats() | st.integers())
def test_fuzz_VectorObject4D(azimuthal, longitudinal, temporal):
    vec = vector.VectorObject4D(
        azimuthal=azimuthal, longitudinal=longitudinal, temporal=temporal
    )
    # assert (vector.obj(**dict(zip(["x", "y", "z", "t"], azimuthal[0], azimuthal[1], longitudinal, temporal)))).tau == 0
    #pytest.approx(ROOT.Math.PxPyPzEVector(azimuthal[0], azimuthal[1], longitudinal, temporal).M())
