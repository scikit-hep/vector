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

# ROOT.Math.RhoZPhiVector constructor arguments to get all the weird cases.
constructor = [
    (0, 0, 0),
    (0, 10, 0),
    (0, -10, 0),
    (1, 0, 0),
    (1, 10, 0),
    (1, -10, 0),
    (1.0, 2.5, 2.0),
    (1, 2.5, 2.0),
    (1, -2.5, 2.0),
]

# Coordinate conversion methods to apply to the VectorObject2D.
coordinate_list = [
    "to_xyz",
    "to_rhophieta",
    "to_rhophitheta",
    "to_rhophiz",
    "to_xyeta",
    "to_xytheta",
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
    assert ROOT.Math.RhoZPhiVector(*constructor).Dot(
        ROOT.Math.RhoZPhiVector(*constructor)
    ) == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
        )().dot(
            getattr(
                vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
            )()
        )
    )


# Run a test that compares ROOT's 'Mag2()' with vector's 'rho2' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Mag2(constructor, coordinates):
    assert ROOT.Math.RhoZPhiVector(*constructor).Mag2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
        )().rho2
    )


# Run a test that compares ROOT's 'R()' with vector's 'rho' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_R(constructor, coordinates):
    assert ROOT.Math.RhoZPhiVector(*constructor).R() == pytest.approx(
        np.sqrt(
            getattr(
                vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
            )().rho2
        )
    )


# Run a test that compares ROOT's 'Phi()' with vector's 'rho' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Phi(constructor, coordinates):
    assert ROOT.Math.RhoZPhiVector(*constructor).Phi() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
        )().phi
    )


# Run a test that compares ROOT's 'RotateX()' with vector's 'rotateX' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_RotateX(constructor, angle, coordinates):
    ref_vec = ROOT.Math.RotationX(angle) * ROOT.Math.RhoZPhiVector(*constructor)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
    )()
    res_vec = vec.rotateX(angle)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)


# Run a test that compares ROOT's 'RotateY()' with vector's 'rotateY' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_RotateY(constructor, angle, coordinates):
    ref_vec = ROOT.Math.RotationY(angle) * ROOT.Math.RhoZPhiVector(*constructor)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
    )()
    res_vec = vec.rotateY(angle)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)


# Run a test that compares ROOT's 'RotateZ()' with vector's 'rotateZ' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_RotateZ(constructor, angle, coordinates):
    ref_vec = ROOT.Math.RotationZ(angle) * ROOT.Math.RhoZPhiVector(*constructor)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
    )()
    res_vec = vec.rotateZ(angle)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)


# Run a test that compares ROOT's 'Unit()' with vector's 'unit' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Unit(constructor, coordinates):
    ref_vec = ROOT.Math.RhoZPhiVector(*constructor).Unit()
    vec = getattr(
        vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
    )()
    res_vec = vec.unit
    assert ref_vec.X() == pytest.approx(res_vec().x)
    assert ref_vec.Y() == pytest.approx(res_vec().y)
    assert ref_vec.Z() == pytest.approx(res_vec().z)


# Run a test that compares ROOT's 'X()' and 'Y()' with vector's 'x' and 'y' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_X_and_Y(constructor, coordinates):
    vec = getattr(
        vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
    )()
    assert ROOT.Math.RhoZPhiVector(*constructor).X() == pytest.approx(
        vec.x
    ) and ROOT.Math.RhoZPhiVector(*constructor).Y() == pytest.approx(vec.y)


# Run a test that compares ROOT's '__add__' with vector's 'add' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_add(constructor, coordinates):
    ref_vec = ROOT.Math.RhoZPhiVector(*constructor).__add__(
        ROOT.Math.RhoZPhiVector(*constructor)
    )
    vec = getattr(
        vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
    )().add(
        getattr(
            vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
        )()
    )
    assert ref_vec.X() == pytest.approx(vec.x)
    assert ref_vec.Y() == pytest.approx(vec.y)
    assert ref_vec.Z() == pytest.approx(vec.z)


# Run a test that compares ROOT's '__sub__' with vector's 'subtract' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_sub(constructor, coordinates):
    ref_vec = ROOT.Math.RhoZPhiVector(*constructor).__sub__(
        ROOT.Math.RhoZPhiVector(*constructor)
    )
    vec1 = getattr(
        vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
    )()
    vec2 = getattr(
        vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
    )()
    res_vec = vec1.subtract(vec2)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)


# Run a test that compares ROOT's '__neg__' with vector's '__neg__' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_neg(constructor, coordinates):
    ref_vec = ROOT.Math.RhoZPhiVector(*constructor).__neg__()
    vec = getattr(
        vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
    )().__neg__
    assert ref_vec.X() == pytest.approx(vec().x)
    assert ref_vec.Y() == pytest.approx(vec().y)
    assert ref_vec.Z() == pytest.approx(vec().z)


# Run a test that compares ROOT's '__mul__' with vector's 'mul' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_mul(constructor, scalar, coordinates):
    ref_vec = ROOT.Math.RhoZPhiVector(*constructor).__mul__(scalar)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
    )().__mul__(scalar)
    assert ref_vec.X() == pytest.approx(vec.x)
    assert ref_vec.Y() == pytest.approx(vec.y)
    assert ref_vec.Z() == pytest.approx(vec.z)


# Run a test that compares ROOT's '__truediv__' with vector's '__truediv__' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_truediv(constructor, scalar, coordinates):
    ref_vec = ROOT.Math.RhoZPhiVector(*constructor).__truediv__(scalar)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
    )().__truediv__(scalar)
    assert ref_vec.X() == pytest.approx(vec.x)
    assert ref_vec.Y() == pytest.approx(vec.y)
    assert ref_vec.Z() == pytest.approx(vec.z)


# Run a test that compares ROOT's '__eq__' with vector's 'isclose' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_eq(constructor, coordinates):
    ref_vec = ROOT.Math.RhoZPhiVector(*constructor).__eq__(
        ROOT.Math.RhoZPhiVector(*constructor)
    )
    vec = getattr(
        vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
    )().isclose(
        getattr(
            vector.obj(**dict(zip(["rho", "z", "phi"], constructor))), coordinates
        )()
    )
    assert ref_vec == vec
