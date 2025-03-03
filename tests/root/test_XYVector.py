# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.
from __future__ import annotations

import numpy as np
import pytest

import vector

# If ROOT is not available, skip these tests.
ROOT = pytest.importorskip("ROOT")

# ROOT.Math.XYVector constructor arguments to get all the weird cases.
constructor = [
    (0, 0),
    (0, 10),
    (0, -10),
    (1, 0),
    (1, 10),
    (1, -10),
    (1.0, 2.5),
    (1, 2.5),
    (1, -2.5),
]

# Coordinate conversion methods to apply to the VectorObject2D.
coordinate_list = [
    "to_xy",
    "to_rhophi",
]


@pytest.fixture(scope="module", params=coordinate_list)
def coordinates(request):
    return request.param


angle_list = [
    0,
    0.0,
    0.7853981633974483,
    -0.7853981633974483,
    1.5707963267948966,
    -1.5707963267948966,
    3.141592653589793,
    -3.141592653589793,
    6.283185307179586,
    -6.283185307179586,
]


@pytest.fixture(scope="module", params=angle_list)
def angle(request):
    return request.param


scalar_list = [
    0,
    -1,
    1.0,
    100000.0000,
    -100000.0000,
]


@pytest.fixture(scope="module", params=scalar_list)
def scalar(request):
    return request.param


# Run a test that compares ROOT's 'Dot()' with vector's 'dot' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Dot(constructor, coordinates):
    assert ROOT.Math.XYVector(*constructor).Dot(
        ROOT.Math.XYVector(*constructor)
    ) == pytest.approx(
        getattr(vector.obj(**dict(zip(["x", "y"], constructor))), coordinates)().dot(
            getattr(vector.obj(**dict(zip(["x", "y"], constructor))), coordinates)()
        ),
        1.0e-6,
        1.0e-6,
    )


# Run a test that compares ROOT's 'Mag2()' with vector's 'rho2' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Mag2(constructor, coordinates):
    assert ROOT.Math.XYVector(*constructor).Mag2() == pytest.approx(
        getattr(vector.obj(**dict(zip(["x", "y"], constructor))), coordinates)().rho2
    )


# Run a test that compares ROOT's 'R()' with vector's 'rho' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_R(constructor, coordinates):
    assert ROOT.Math.XYVector(*constructor).R() == pytest.approx(
        np.sqrt(
            getattr(
                vector.obj(**dict(zip(["x", "y"], constructor))), coordinates
            )().rho2
        )
    )


# Run a test that compares ROOT's 'Phi()' with vector's 'rho' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Phi(constructor, coordinates):
    assert ROOT.Math.XYVector(*constructor).Phi() == pytest.approx(
        getattr(vector.obj(**dict(zip(["x", "y"], constructor))), coordinates)().phi
    )


# Run a test that compares ROOT's 'Rotate()' with vector's 'rotateZ' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Rotate(constructor, angle, coordinates):
    ref_vec = ROOT.Math.XYVector(*constructor)
    ref_vec.Rotate(angle)
    vec = getattr(vector.obj(**dict(zip(["x", "y"], constructor))), coordinates)()
    res_vec = vec.rotateZ(angle)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)


# Run a test that compares ROOT's 'Unit()' with vector's 'unit' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Unit(constructor, coordinates):
    # FIXME: if x == 0 and y == 0
    # assert 0.0 == 1.0 ± 1.0e-06
    if constructor[0] != 0 and constructor[1] != 0:
        ref_vec = ROOT.Math.XYVector(*constructor).Unit()
        vec = getattr(vector.obj(**dict(zip(["x", "y"], constructor))), coordinates)()
        res_vec = vec.unit
        assert ref_vec.X() == pytest.approx(res_vec().x)
        assert ref_vec.Y() == pytest.approx(res_vec().y)


# Run a test that compares ROOT's 'X()' and 'Y()' with vector's 'x' and 'y' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_X_and_Y(constructor, coordinates):
    vec = getattr(vector.obj(**dict(zip(["x", "y"], constructor))), coordinates)()
    assert ROOT.Math.XYVector(*constructor).X() == pytest.approx(
        vec.x
    ) and ROOT.Math.XYVector(*constructor).Y() == pytest.approx(vec.y)


# Run a test that compares ROOT's '__add__' with vector's 'add' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_add(constructor, coordinates):
    ref_vec = ROOT.Math.XYVector(*constructor).__add__(ROOT.Math.XYVector(*constructor))
    vec = getattr(vector.obj(**dict(zip(["x", "y"], constructor))), coordinates)().add(
        getattr(vector.obj(**dict(zip(["x", "y"], constructor))), coordinates)()
    )
    assert ref_vec.X() == pytest.approx(vec.x)
    assert ref_vec.Y() == pytest.approx(vec.y)


# Run a test that compares ROOT's '__sub__' with vector's 'subtract' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_sub(constructor, coordinates):
    ref_vec = ROOT.Math.XYVector(*constructor).__sub__(ROOT.Math.XYVector(*constructor))
    vec1 = getattr(vector.obj(**dict(zip(["x", "y"], constructor))), coordinates)()
    vec2 = getattr(vector.obj(**dict(zip(["x", "y"], constructor))), coordinates)()
    res_vec = vec1.subtract(vec2)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)


# Run a test that compares ROOT's '__neg__' with vector's '__neg__' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_neg(constructor, coordinates):
    ref_vec = ROOT.Math.XYVector(*constructor).__neg__()
    vec = getattr(
        vector.obj(**dict(zip(["x", "y"], constructor))), coordinates
    )().__neg__
    assert ref_vec.X() == pytest.approx(vec().x)
    assert ref_vec.Y() == pytest.approx(vec().y)


# Run a test that compares ROOT's '__mul__' with vector's 'mul' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_mul(constructor, scalar, coordinates):
    ref_vec = ROOT.Math.XYVector(*constructor).__mul__(scalar)
    vec = getattr(
        vector.obj(**dict(zip(["x", "y"], constructor))), coordinates
    )().__mul__(scalar)
    assert ref_vec.X() == pytest.approx(
        vec.x, 1.0e-6, 1.0e-6
    ) and ref_vec.Y() == pytest.approx(vec.y, 1.0e-6, 1.0e-6)


# Run a test that compares ROOT's '__truediv__' with vector's '__truediv__' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_truediv(constructor, scalar, coordinates):
    if scalar != 0:
        ref_vec = ROOT.Math.XYVector(*constructor).__truediv__(scalar)
        vec = getattr(
            vector.obj(**dict(zip(["x", "y"], constructor))), coordinates
        )().__truediv__(scalar)
        assert ref_vec.X() == pytest.approx(vec.x)
        assert ref_vec.Y() == pytest.approx(vec.y)


# Run a test that compares ROOT's '__eq__' with vector's 'isclose' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_eq(constructor, coordinates):
    ref_vec = ROOT.Math.XYVector(*constructor).__eq__(ROOT.Math.XYVector(*constructor))
    vec = getattr(
        vector.obj(**dict(zip(["x", "y"], constructor))), coordinates
    )().isclose(
        getattr(vector.obj(**dict(zip(["x", "y"], constructor))), coordinates)()
    )
    assert ref_vec == vec
