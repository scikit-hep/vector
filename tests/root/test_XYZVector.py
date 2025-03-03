# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.
from __future__ import annotations

import pytest

import vector

# If ROOT is not available, skip these tests.
ROOT = pytest.importorskip("ROOT")

# ROOT.Math.XYZVector constructor arguments to get all the weird cases.
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
    assert ROOT.Math.XYZVector(*constructor).Dot(
        ROOT.Math.XYZVector(*constructor)
    ) == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
        )().dot(
            getattr(
                vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
            )()
        )
    )


# Run a test that compares ROOT's 'Cross()' with vector's 'cross' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Cross(constructor, coordinates):
    ref_vec = ROOT.Math.XYZVector(*constructor).Cross(ROOT.Math.XYZVector(*constructor))
    vec = getattr(
        vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
    )().cross(
        getattr(vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates)()
    )
    assert (
        ref_vec.X()
        == pytest.approx(
            vec.x,
            1.0e-6,
            1.0e-6,
        )
        and ref_vec.Y()
        == pytest.approx(
            vec.y,
            1.0e-6,
            1.0e-6,
        )
        and ref_vec.Z()
        == pytest.approx(
            vec.z,
            1.0e-6,
            1.0e-6,
        )
    )


# Run a test that compares ROOT's 'Mag2()' with vector's 'mag2' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Mag2(constructor, coordinates):
    ref_vec = ROOT.Math.XYZVector(*constructor)
    vec = getattr(vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates)()
    assert (
        pytest.approx(vec.x) == ref_vec.X()
        and pytest.approx(vec.y) == ref_vec.Y()
        and pytest.approx(vec.z) == ref_vec.Z()
    )

    assert ref_vec.Mag2() == pytest.approx(vec.mag2, 1.0e-6, 1.0e-6)


# Run a test that compares ROOT's 'Mag()' with vector's 'mag' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Mag(constructor, coordinates):
    assert ROOT.Math.XYZVector(*constructor).R() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
        )().mag,
        1.0e-6,
        1.0e-6,
    )


# Run a test that compares ROOT's 'Perp2()' with vector's 'rho2' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Perp2(constructor, coordinates):
    assert ROOT.Math.XYZVector(*constructor).Perp2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
        )().rho2,
        1.0e-6,
        1.0e-6,
    )


# Run a test that compares ROOT's 'Perp()' with vector's 'rho' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Perp(constructor, coordinates):
    assert ROOT.Math.sqrt(ROOT.Math.XYZVector(*constructor).Perp2()) == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
        )().rho,
        1.0e-6,
        1.0e-6,
    )


# Run a test that compares ROOT's 'Phi()' with vector's 'rho' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Phi(constructor, coordinates):
    assert ROOT.Math.XYZVector(*constructor).Phi() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
        )().phi
    )


# Run a test that compares ROOT's 'Eta()' with vector's 'eta' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Eta(constructor, coordinates):
    assert ROOT.Math.XYZVector(*constructor).Eta() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
        )().eta,
        1.0e-6,
        1.0e-6,
    )


# Run a test that compares ROOT's 'Theta()' with vector's 'theta' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Theta(constructor, coordinates):
    assert ROOT.Math.XYZVector(*constructor).Theta() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
        )().theta
    )


# Run a test that compares ROOT's 'RotateX()' with vector's 'rotateX' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_RotateX(constructor, angle, coordinates):
    ref_vec = ROOT.Math.RotationX(angle) * ROOT.Math.XYZVector(*constructor)
    vec = getattr(vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates)()
    res_vec = vec.rotateX(angle)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)


# Run a test that compares ROOT's 'RotateY()' with vector's 'rotateY' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_RotateY(constructor, angle, coordinates):
    ref_vec = ROOT.Math.RotationY(angle) * ROOT.Math.XYZVector(*constructor)
    vec = getattr(vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates)()
    res_vec = vec.rotateY(angle)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)


# Run a test that compares ROOT's 'RotateZ()' with vector's 'rotateZ' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_RotateZ(constructor, angle, coordinates):
    ref_vec = ROOT.Math.RotationZ(angle) * ROOT.Math.XYZVector(*constructor)
    vec = getattr(vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates)()
    res_vec = vec.rotateZ(angle)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)


# Run a test that compares ROOT's 'Unit()' with vector's 'unit' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Unit(constructor, coordinates):
    ref_vec = ROOT.Math.XYZVector(*constructor).Unit()
    vec = getattr(vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates)()
    res_vec = vec.unit
    assert ref_vec.X() == pytest.approx(res_vec().x)
    assert ref_vec.Y() == pytest.approx(res_vec().y)
    assert ref_vec.Z() == pytest.approx(res_vec().z)


# Run a test that compares ROOT's 'X()' and 'Y()' with vector's 'x' and 'y' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_X_Y_Z(constructor, coordinates):
    vec = getattr(vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates)()
    assert (
        ROOT.Math.XYZVector(*constructor).X() == pytest.approx(vec.x)
        and ROOT.Math.XYZVector(*constructor).Y() == pytest.approx(vec.y)
        and ROOT.Math.XYZVector(*constructor).Z() == pytest.approx(vec.z)
    )


# Run a test that compares ROOT's '__add__' with vector's 'add' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_add(constructor, coordinates):
    ref_vec = ROOT.Math.XYZVector(*constructor).__add__(
        ROOT.Math.XYZVector(*constructor)
    )
    vec = getattr(
        vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
    )().add(
        getattr(vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates)()
    )
    assert ref_vec.X() == pytest.approx(vec.x)
    assert ref_vec.Y() == pytest.approx(vec.y)
    assert ref_vec.Z() == pytest.approx(vec.z)


# Run a test that compares ROOT's '__sub__' with vector's 'subtract' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_sub(constructor, coordinates):
    ref_vec = ROOT.Math.XYZVector(*constructor).__sub__(
        ROOT.Math.XYZVector(*constructor)
    )
    vec1 = getattr(vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates)()
    vec2 = getattr(vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates)()
    res_vec = vec1.subtract(vec2)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)


# Run a test that compares ROOT's '__neg__' with vector's '__neg__' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_neg(constructor, coordinates):
    ref_vec = ROOT.Math.XYZVector(*constructor).__neg__()
    vec = getattr(
        vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
    )().__neg__
    assert ref_vec.X() == pytest.approx(vec().x)
    assert ref_vec.Y() == pytest.approx(vec().y)
    assert ref_vec.Z() == pytest.approx(vec().z)


# Run a test that compares ROOT's '__mul__' with vector's 'mul' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_mul(constructor, scalar, coordinates):
    ref_vec = ROOT.Math.XYZVector(*constructor).__mul__(scalar)
    vec = getattr(
        vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
    )().__mul__(scalar)
    assert ref_vec.X() == pytest.approx(vec.x, 1.0e-6, 1.0e-6)
    assert ref_vec.Y() == pytest.approx(vec.y, 1.0e-6, 1.0e-6)
    assert ref_vec.Z() == pytest.approx(vec.z, 1.0e-6, 1.0e-6)


# Run a test that compares ROOT's '__truediv__' with vector's '__truediv__' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_truediv(constructor, scalar, coordinates):
    # FIXME
    if scalar != 0:
        ref_vec = ROOT.Math.XYZVector(*constructor).__truediv__(scalar)
        vec = getattr(
            vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
        )().__truediv__(scalar)
        assert ref_vec.X() == pytest.approx(vec.x)
        assert ref_vec.Y() == pytest.approx(vec.y)
        assert ref_vec.Z() == pytest.approx(vec.z)


# Run a test that compares ROOT's '__eq__' with vector's 'isclose' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_eq(constructor, coordinates):
    ref_vec = ROOT.Math.XYZVector(*constructor).__eq__(
        ROOT.Math.XYZVector(*constructor)
    )
    vec = getattr(
        vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates
    )().isclose(
        getattr(vector.obj(**dict(zip(["x", "y", "z"], constructor))), coordinates)()
    )
    assert ref_vec == vec
