# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest
from hypothesis import given
from hypothesis import strategies as st

import vector

# If ROOT is not available, skip these tests.
ROOT = pytest.importorskip("ROOT")

# ROOT.Math.Polar3DVector constructor arguments to get all the weird cases.
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
# rho = r*sin(theta)
@pytest.mark.parametrize("constructor", constructor)
def test_Dot(constructor, coordinates):
    assert ROOT.Math.Polar3DVector(*constructor).Dot(
        ROOT.Math.Polar3DVector(*constructor)
    ) == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
        )().dot(
            getattr(
                vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))),
                coordinates,
            )()
        )
    )


# Run the same tests within hypothesis
@given(
    constructor1=st.tuples(
        st.floats(min_value=-10e7, max_value=10e7),
        st.floats(min_value=-10e7, max_value=10e7),
        st.floats(min_value=-10e7, max_value=10e7),
    )
    | st.tuples(
        st.integers(min_value=-10e7, max_value=10e7),
        st.integers(min_value=-10e7, max_value=10e7),
        st.integers(min_value=-10e7, max_value=10e7),
    ),
    constructor2=st.tuples(
        st.floats(min_value=-10e7, max_value=10e7),
        st.floats(min_value=-10e7, max_value=10e7),
        st.floats(min_value=-10e7, max_value=10e7),
    )
    | st.tuples(
        st.integers(min_value=-10e7, max_value=10e7),
        st.integers(min_value=-10e7, max_value=10e7),
        st.integers(min_value=-10e7, max_value=10e7),
    ),
)
def test_fuzz_Dot(constructor1, constructor2, coordinates):
    assert ROOT.Math.Polar3DVector(*constructor1).Dot(
        ROOT.Math.Polar3DVector(*constructor2)
    ) == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor1))), coordinates
        )().dot(
            getattr(
                vector.obj(**dict(zip(["rho", "theta", "phi"], constructor2))),
                coordinates,
            )()
        )
    )


# Run a test that compares ROOT's 'Cross()' with vector's 'cross' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Cross(constructor, coordinates):
    ref_vec = ROOT.Math.Polar3DVector(*constructor).Cross(
        ROOT.Math.Polar3DVector(*constructor)
    )
    vec = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
    )().cross(
        getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
        )()
    )
    assert (
        ref_vec.Rho()
        == pytest.approx(
            vec.rho,
            1.0e-6,
            1.0e-6,
        )
        and ref_vec.Theta()
        == pytest.approx(
            vec.theta,
            1.0e-6,
            1.0e-6,
        )
        and ref_vec.Phi()
        == pytest.approx(
            vec.phi,
            1.0e-6,
            1.0e-6,
        )
    )


# Run the same tests within hypothesis
@given(
    constructor1=st.tuples(
        st.floats(min_value=-10e7, max_value=10e7),
        st.floats(min_value=-10e7, max_value=10e7),
        st.floats(min_value=-10e7, max_value=10e7),
    )
    | st.tuples(
        st.integers(min_value=-10e7, max_value=10e7),
        st.integers(min_value=-10e7, max_value=10e7),
        st.integers(min_value=-10e7, max_value=10e7),
    ),
    constructor2=st.tuples(
        st.floats(min_value=-10e7, max_value=10e7),
        st.floats(min_value=-10e7, max_value=10e7),
        st.floats(min_value=-10e7, max_value=10e7),
    )
    | st.tuples(
        st.integers(min_value=-10e7, max_value=10e7),
        st.integers(min_value=-10e7, max_value=10e7),
        st.integers(min_value=-10e7, max_value=10e7),
    ),
)
def test_fuzz_Cross(constructor1, constructor2, coordinates):
    ref_vec = ROOT.Math.Polar3DVector(*constructor1).Cross(
        ROOT.Math.Polar3DVector(*constructor2)
    )
    vec = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor1))), coordinates
    )().cross(
        getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor2))), coordinates
        )()
    )
    assert (
        ref_vec.Rho()
        == pytest.approx(
            vec.rho,
            1.0e-6,
            1.0e-6,
        )
        and ref_vec.Theta()
        == pytest.approx(
            vec.theta,
            1.0e-6,
            1.0e-6,
        )
        and ref_vec.Phi()
        == pytest.approx(
            vec.phi,
            1.0e-6,
            1.0e-6,
        )
    )


# Run a test that compares ROOT's 'Mag2()' with vector's 'rho2' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Mag2(constructor, coordinates):
    assert ROOT.Math.Polar3DVector(*constructor).Mag2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
        )().mag2
    )


# Run a test that compares ROOT's 'Mag()' with vector's 'mag' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_R(constructor, coordinates):
    assert ROOT.Math.sqrt(
        ROOT.Math.Polar3DVector(*constructor).Mag2()
    ) == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))),
            coordinates,
        )().mag
    )


# Run a test that compares ROOT's 'Perp2()' with vector's 'rho2' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Perp2(constructor, coordinates):
    assert ROOT.Math.Polar3DVector(*constructor).Perp2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
        )().rho2
    )


# Run a test that compares ROOT's 'Rho()' with vector's 'rho' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Rho(constructor, coordinates):
    assert ROOT.Math.Polar3DVector(*constructor).Rho() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))),
            coordinates,
        )().rho
    )


# Run a test that compares ROOT's 'Phi()' with vector's 'phi' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Phi(constructor, coordinates):
    assert ROOT.Math.Polar3DVector(*constructor).Phi() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
        )().phi
    )


# Run a test that compares ROOT's 'Eta()' with vector's 'eta' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def Eta(constructor, coordinates):
    assert ROOT.Math.Polar3DVector(*constructor).Eta() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
        )().eta
    )


# Run a test that compares ROOT's 'Theta()' with vector's 'theta' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Theta(constructor, coordinates):
    assert ROOT.Math.Polar3DVector(*constructor).Theta() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
        )().theta
    )


# Run a test that compares ROOT's 'RotateX()' with vector's 'rotateX' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_RotateX(constructor, angle, coordinates):
    ref_vec = ROOT.Math.RotationX(angle) * ROOT.Math.Polar3DVector(*constructor)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
    )()
    res_vec = vec.rotateX(angle)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)


# Run a test that compares ROOT's 'RotateY()' with vector's 'rotateY' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_RotateY(constructor, angle, coordinates):
    ref_vec = ROOT.Math.RotationY(angle) * ROOT.Math.Polar3DVector(*constructor)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
    )()
    res_vec = vec.rotateY(angle)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)


# Run a test that compares ROOT's 'RotateZ()' with vector's 'rotateZ' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_RotateZ(constructor, angle, coordinates):
    ref_vec = ROOT.Math.RotationZ(angle) * ROOT.Math.Polar3DVector(*constructor)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
    )()
    res_vec = vec.rotateZ(angle)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)


# Run a test that compares ROOT's 'RotateAxes' with vector's 'rotate_axes' for all cases.
def test_RotateAxes(constructor, angle, coordinates):
    ref_vec = ROOT.Math.Polar3DVector(*constructor)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
    )()
    # FIXME: rotate_axis
    assert (
        ref_vec.Rho()
        == pytest.approx(
            vec.rho,
            1.0e-6,
            1.0e-6,
        )
        and ref_vec.Theta()
        == pytest.approx(
            vec.theta,
            1.0e-6,
            1.0e-6,
        )
        and ref_vec.Phi()
        == pytest.approx(
            vec.phi,
            1.0e-6,
            1.0e-6,
        )
    )


# Run a test that compares ROOT's 'Unit()' with vector's 'unit' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Unit(constructor, coordinates):
    ref_vec = ROOT.Math.Polar3DVector(*constructor).Unit()
    vec = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
    )()
    res_vec = vec.unit
    assert ref_vec.X() == pytest.approx(res_vec().x)
    assert ref_vec.Y() == pytest.approx(res_vec().y)
    assert ref_vec.Z() == pytest.approx(res_vec().z)


# Run a test that compares ROOT's 'X()' and 'Y()' with vector's 'x' and 'y' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_X_Y_Z(constructor, coordinates):
    vec = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
    )()
    assert (
        ROOT.Math.Polar3DVector(*constructor).X() == pytest.approx(vec.x)
        and ROOT.Math.Polar3DVector(*constructor).Y() == pytest.approx(vec.y)
        and ROOT.Math.Polar3DVector(*constructor).Z() == pytest.approx(vec.z)
    )


# Run a test that compares ROOT's '__add__' with vector's 'add' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_add(constructor, coordinates):
    ref_vec = ROOT.Math.Polar3DVector(*constructor).__add__(
        ROOT.Math.Polar3DVector(*constructor)
    )
    vec = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
    )().add(
        getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
        )()
    )
    assert ref_vec.X() == pytest.approx(vec.x)
    assert ref_vec.Y() == pytest.approx(vec.y)
    assert ref_vec.Z() == pytest.approx(vec.z)


# Run a test that compares ROOT's '__sub__' with vector's 'subtract' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_sub(constructor, coordinates):
    ref_vec = ROOT.Math.Polar3DVector(*constructor).__sub__(
        ROOT.Math.Polar3DVector(*constructor)
    )
    vec1 = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
    )()
    vec2 = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
    )()
    res_vec = vec1.subtract(vec2)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)


# Run a test that compares ROOT's '__neg__' with vector's '__neg__' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_neg(constructor, coordinates):
    ref_vec = ROOT.Math.Polar3DVector(*constructor).__neg__()
    vec = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
    )().__neg__
    assert ref_vec.X() == pytest.approx(vec().x)
    assert ref_vec.Y() == pytest.approx(vec().y)
    assert ref_vec.Z() == pytest.approx(vec().z)


# Run a test that compares ROOT's '__mul__' with vector's 'mul' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_mul(constructor, scalar, coordinates):
    ref_vec = ROOT.Math.Polar3DVector(*constructor).__mul__(scalar)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
    )().__mul__(scalar)
    assert ref_vec.X() == pytest.approx(vec.x)
    assert ref_vec.Y() == pytest.approx(vec.y)
    assert ref_vec.Z() == pytest.approx(vec.z)


# Run a test that compares ROOT's '__truediv__' with vector's '__truediv__' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_truediv(constructor, scalar, coordinates):
    # FIXME
    if scalar != 0:
        ref_vec = ROOT.Math.Polar3DVector(*constructor).__truediv__(scalar)
        vec = getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
        )().__truediv__(scalar)
        assert ref_vec.X() == pytest.approx(vec.x)
        assert ref_vec.Y() == pytest.approx(vec.y)
        assert ref_vec.Z() == pytest.approx(vec.z)


# Run a test that compares ROOT's '__eq__' with vector's 'isclose' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_eq(constructor, coordinates):
    ref_vec = ROOT.Math.Polar3DVector(*constructor).__eq__(
        ROOT.Math.Polar3DVector(*constructor)
    )
    vec = getattr(
        vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
    )().isclose(
        getattr(
            vector.obj(**dict(zip(["rho", "theta", "phi"], constructor))), coordinates
        )()
    )
    assert ref_vec == vec
