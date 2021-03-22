# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest
from hypothesis import given, strategies as st

import vector

# If ROOT is not available, skip these tests.
ROOT = pytest.importorskip("ROOT")

# ROOT.Math.PxPyPzEVector constructor arguments to get all the weird cases.
constructor = [
    (0, 0, 0, 0),
#    (0, 0, 1, 0), # theta == 0.0
#    (0, 0, -1, 0),
    (0, 0, 0, 10),
    (0, 0, 0, -10),
    (1, 2, 3, 0),
    (1, 2, 3, 10),
    (1, 2, 3, -10),
    (1., 2., 3., 2.5),
    (1, 2, 3, 2.5),
    (1, 2, 3, -2.5),
]

# Coordinate conversion methods to apply to the VectorObject4D.
coordinate_list = [
    "to_xyzt",
    "to_xythetat", # may fail for constructor2
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

# Run a test that compares ROOT's 'M2()' with vector's 'tau2' for all cases.
# Mass2 is our tau2 (or mass2 if it's a momentum vector and has kinematic synonyms)
@pytest.mark.parametrize("constructor", constructor)
def test_M2(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).M2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().tau2
    )

# Run the same tests within hypothesis
<<<<<<< HEAD
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7)) | st.tuples(st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7)))
=======
@given(constructor=st.tuples(st.floats(), st.floats(), st.floats(), st.floats()) | st.tuples(st.integers(), st.integers(), st.integers(), st.integers()))
>>>>>>> test fixture and hypothesis
def test_fuzz_M2(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).M2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().tau2
    )

# Run a test that compares ROOT's 'M()' with vector's 'tau' for all cases.
# Mass is tau (or mass)
@pytest.mark.parametrize("constructor", constructor)
def test_M(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).M() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().tau
    )

# Run the same tests within hypothesis
<<<<<<< HEAD
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7)) | st.tuples(st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7)))
=======
@given(constructor=st.tuples(st.floats(), st.floats(), st.floats(), st.floats()) | st.tuples(st.integers(), st.integers(), st.integers(), st.integers()))
>>>>>>> test fixture and hypothesis
def test_fuzz_M(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).M() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().tau
    )

# Run a test that compares ROOT's 'Dot()' with vector's 'dot' for all cases.
# Dot
@pytest.mark.parametrize("constructor", constructor)
def test_Dot(constructor, coordinates):
<<<<<<< HEAD
    v1 = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )()
    v2 = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )()
    assert ROOT.Math.PxPyPzEVector(*constructor).Dot(ROOT.Math.PxPyPzEVector(*constructor)) == pytest.approx(
        v1.dot(v2)
    )

# Run the same test within hypothesis
@given(constructor1=st.tuples(st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7))
                    | st.tuples(st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7)),
       constructor2=st.tuples(st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7))
                    | st.tuples(st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7)))
def test_fizz_Dot(constructor1, constructor2, coordinates):
    v1 = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor1))), coordinates
    )()
    v2 = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor2))), coordinates
    )()
    assert ROOT.Math.PxPyPzEVector(*constructor1).Dot(ROOT.Math.PxPyPzEVector(*constructor2)) == pytest.approx(
        v1.dot(v2)
    )

# Run a test that compares ROOT's 'Mt2()' with vector's 'mt2' for all cases.
# Mt2 same for transverse mass: it's only on momentum vectors
@pytest.mark.parametrize("constructor", constructor)
def test_Mt2(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).mt2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().mt2
    )

# Run a test that compares ROOT's 'Mt()' with vector's 'mt' for all cases.
# Mt
@pytest.mark.parametrize("constructor", constructor)
def test_Mt(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).mt() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().mt
    )

# Run a test that compares ROOT's 'Mag2()' with vector's 'mag2' for all cases.
# P2 is our mag2 (ROOT's 4D mag2 is the dot product with itself, what we call tau or mass)
@pytest.mark.parametrize("constructor", constructor)
def test_Mag2(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).P2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().mag2
    )

# Run a test that compares ROOT's 'Mag()' with vector's 'mag' for all cases.
# P is our mag (same deal)
@pytest.mark.parametrize("constructor", constructor)
def test_P2(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).P() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().mag
    )

# Run a test that compares ROOT's 'Minus()' with vector's 'mag' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Minus(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).Minus() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().minus
=======
    assert ROOT.Math.PxPyPzEVector(*constructor).Dot(ROOT.Math.PxPyPzEVector(*constructor)) == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().dot(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )())
    )

# Run the same test within hypothesis
@given(constructor1=st.tuples(st.floats(), st.floats(), st.floats(), st.floats())
                    | st.tuples(st.integers(), st.integers(), st.integers(), st.integers()),
       constructor2=st.tuples(st.floats(), st.floats(), st.floats(), st.floats())
                    | st.tuples(st.integers(), st.integers(), st.integers(), st.integers()))
def test_fizz_Dot(constructor1, constructor2, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor1).Dot(ROOT.Math.PxPyPzEVector(*constructor2)) == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor1))), coordinates
        )().dot(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor2))), coordinates
        )())
>>>>>>> test fixture and hypothesis
    )
