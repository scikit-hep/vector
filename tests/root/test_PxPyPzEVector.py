# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest

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
coordinates = [
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


# Run a test that compares ROOT's 'M2()' with vector's 'tau2' for all cases.
@pytest.mark.parametrize("constructor", constructor)
@pytest.mark.parametrize("coordinates", coordinates)
def test_M2(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).M2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().tau2
    )


# Run a test that compares ROOT's 'M()' with vector's 'tau' for all cases.
@pytest.mark.parametrize("constructor", constructor)
@pytest.mark.parametrize("coordinates", coordinates)
def test_M(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).M() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().tau
    )


# etc.
