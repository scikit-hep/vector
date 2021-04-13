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
    (0, 0, 0, 10),
    (0, 0, 0, -10),
    (1, 2, 3, 0),
    (1, 2, 3, 10),
    (1, 2, 3, -10),
    (1, 2, 3, 2.5),
    (1, 2, 3, -2.5),
]

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
@given(constructor=st.tuples(st.floats(), st.floats(), st.floats(), st.floats()) | st.tuples(st.integers(), st.integers(), st.integers(), st.integers()))
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
@given(constructor=st.tuples(st.floats(), st.floats(), st.floats(), st.floats()) | st.tuples(st.integers(), st.integers(), st.integers(), st.integers()))
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
    )

# # Run a test that compares ROOT's 'Mt2()' with vector's 'mt2' for all cases.
# # Mt2 same for transverse mass: it's only on momentum vectors
# @pytest.mark.parametrize("constructor", constructor)
# def test_Mt2(constructor, coordinates):
#     assert ROOT.Math.PxPyPzEVector(*constructor).Mt2() == pytest.approx(
#         getattr(
#             vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
#         )().mt2
#     )
#
# # Run a test that compares ROOT's 'Mt()' with vector's 'mt' for all cases.
# # Mt
# @pytest.mark.parametrize("constructor", constructor)
# def test_Mt(constructor, coordinates):
#     assert ROOT.Math.PxPyPzEVector(*constructor).Mt() == pytest.approx(
#         getattr(
#             vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
#         )().mt
#     )

# # Run a test that compares ROOT's 'Mag2()' with vector's 'mag2' for all cases.
# # P2 is our mag2 (ROOT's 4D mag2 is the dot product with itself, what we call tau or mass)
# @pytest.mark.parametrize("constructor", constructor)
# def test_Mag2(constructor, coordinates):
#     assert ROOT.Math.PxPyPzEVector(*constructor).Mag2() == pytest.approx(
#         getattr(
#             vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
#         )().mag2
#     )

# # Run a test that compares ROOT's 'Mag()' with vector's 'mag' for all cases.
# # P is our mag (same deal)
# @pytest.mark.parametrize("constructor", constructor)
# def test_Mag(constructor, coordinates):
#     assert ROOT.Math.PxPyPzEVector(*constructor).Mag() == pytest.approx(
#         getattr(
#             vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
#         )().mag
#     )

# 'Minus()'
# 'Mt2()'

  # Perp2/rho2
  # Perp/rho
  # Phi
  # Eta
  # Theta

  # Rapidity
  # Beta (scalar)
  # Gamma (scalar)
  # BoostToCM is our beta3 (it doesn't boost: it returns a velocity vector for which c=1)
  # ColinearRapidity (we don't have an equivalent, but perhaps we should)
  # Et2 to have a method for transverse energy, you have to construct a vector.obj with momentum coordinates
  # Et

  # isLightlike/is_lightlike
  # isSpacelike/is_spacelike
  # isTimelike/is_timelike
  # ROOT's rotateX, rotateY, rotateZ, and rotate_axis are in its VectorUtil namespace (see below)
  # so are the boosts
  # Unit
  # X, Y, Z, T
  # __add__ (addition by a vector)
  # __sub__ (subtraction by a vector)
  # __neg__ (unary negation of a vector)
  # __mul__ (multiplication by a scalar)
  # __truediv__ (division by a scalar)
  # __eq__ (vector equality), but since you're going through different coordinate systems, use isclose
# etc.
