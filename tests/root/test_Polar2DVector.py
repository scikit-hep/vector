# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest
from hypothesis import given, strategies as st

import numpy as np

import vector

# If ROOT is not available, skip these tests.
ROOT = pytest.importorskip("ROOT")

# ROOT.Math.Polar2DVector constructor arguments to get all the weird cases.
# r > 0 and phi from 0 to 360 deg?
constructor = [
    (0, 0),
    (0, 10),
    (1, 0),
    (10, 0),
    (1, 10),
    (10., 10),
    (1., 2.5),
    (1, 2.5),
    (1, 6.283185307179586),
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
    0.25*np.pi,
    -0.25*np.pi,
    0.5*np.pi,
    -0.5*np.pi,
    np.pi,
    -np.pi,
    2*np.pi,
    -2*np.pi,
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
    assert ROOT.Math.Polar2DVector(*constructor).Dot(ROOT.Math.Polar2DVector(*constructor)) == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
        )().dot(getattr(
            vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
        )())
    )


# Run the same tests within hypothesis
@given(constructor1=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                              st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)),
       constructor2=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                              st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)))
def test_fuzz_Dot(constructor1, constructor2, coordinates):
    assert ROOT.Math.Polar2DVector(*constructor1).Dot(ROOT.Math.Polar2DVector(*constructor2)) == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "phi"], constructor1))), coordinates
        )().dot(getattr(
            vector.obj(**dict(zip(["rho", "phi"], constructor2))), coordinates
        )()), rel=1e-6, abs=1e-6
    )


# Run a test that compares ROOT's 'Mag2()' with vector's 'rho2' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Mag2(constructor, coordinates):
    assert ROOT.Math.Polar2DVector(*constructor).Mag2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
        )().rho2
    )


# Run the same tests within hypothesis
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                             st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)))
def test_fuzz_Mag2(constructor, coordinates):
    assert ROOT.Math.Polar2DVector(*constructor).Mag2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
        )().rho2
    )


# Run a test that compares ROOT's 'R()' with vector's 'rho' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Mag(constructor, coordinates):
    assert ROOT.Math.sqrt(ROOT.Math.Polar2DVector(*constructor).Mag2()) == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
        )().rho
    )


# Run the same tests within hypothesis
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                             st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)))
def test_fuzz_Mag(constructor, coordinates):
    assert ROOT.Math.sqrt(ROOT.Math.Polar2DVector(*constructor).Mag2()) == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
        )().rho
    )


# Run a test that compares ROOT's 'Phi()' with vector's 'rho' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Phi(constructor, coordinates):
    assert ROOT.Math.Polar2DVector(*constructor).Phi() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
        )().phi
    )

# Run the same tests within hypothesis
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                             st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)))
def test_fuzz_Phi(constructor, coordinates):
    assert ROOT.Math.Polar2DVector(*constructor).Phi() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
        )().phi
    )

# Run a test that compares ROOT's 'Rotate()' with vector's 'rotateZ' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Rotate(constructor, angle, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor)
    ref_vec.Rotate(angle)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )().rotateZ(angle)
    res_vec = vec.rotateZ(angle)
    assert (ref_vec.R() == pytest.approx(res_vec.rho) and
            ref_vec.Phi() == pytest.approx(res_vec.phi))


# Run the same tests within hypothesis
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                             st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)),
       angle=st.floats(min_value=-10e7, max_value=10e7)
        |  st.integers(min_value=-10e7, max_value=10e7))
def test_fuzz_Rotate(constructor, angle, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor)
    ref_vec.Rotate(angle)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )()
    res_vec = vec.rotateZ(angle)
    assert (ref_vec.R() == pytest.approx(res_vec.rho)  and
            ref_vec.Phi() == pytest.approx(res_vec.phi))


# Run a test that compares ROOT's 'Unit()' with vector's 'unit' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Unit(constructor, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor).Unit()
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )()
    res_vec = vec.unit
    assert (ref_vec.R() == pytest.approx(res_vec().rho)  and
            ref_vec.Phi() == pytest.approx(res_vec().phi))


# Run the same tests within hypothesis
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                             st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)))
def test_fuzz_Unit(constructor, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor).Unit()
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )()
    res_vec = vec.unit
    assert (ref_vec.R() == pytest.approx(res_vec().rho)  and
            ref_vec.Phi() == pytest.approx(res_vec().phi))


# Run a test that compares ROOT's 'X()' and 'Y()' with vector's 'x' and 'y' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_X_and_Y(constructor, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )()
    assert (ref_vec.X() == pytest.approx(vec.x)  and
            ref_vec.Y() == pytest.approx(vec.y))


# Run the same tests within hypothesis
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                             st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)))
def test_X_and_Y(constructor, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )()
    assert (ref_vec.X() == pytest.approx(vec.x)  and
            ref_vec.Y() == pytest.approx(vec.y))


# Run a test that compares ROOT's '__add__' with vector's 'add' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_add(constructor, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor).__add__(ROOT.Math.Polar2DVector(*constructor))
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )().add(getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates)())
    assert (ref_vec.R() == pytest.approx(vec.rho)  and
            ref_vec.Phi() == pytest.approx(vec.phi))


# Run the same tests within hypothesis
@given(constructor1=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                              st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)),
       constructor2=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                              st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)))
def test_fuzz_add(constructor1, constructor2, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor1).__add__(ROOT.Math.Polar2DVector(*constructor2))
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor1))), coordinates
    )().add(getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor2))), coordinates)())
    assert (ref_vec.R() == pytest.approx(vec.rho)  and
            ref_vec.Phi() == pytest.approx(vec.phi))


# Run a test that compares ROOT's '__sub__' with vector's 'subtract' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_sub(constructor, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor).__sub__(ROOT.Math.Polar2DVector(*constructor))
    vec1 = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )()
    vec2 = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )()
    res_vec = vec1.subtract(vec2)
    assert (ref_vec.R() == pytest.approx(res_vec.rho)  and
            ref_vec.Phi() == pytest.approx(res_vec.phi))


# Run the same tests within hypothesis
@given(constructor1=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                              st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)),
       constructor2=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                              st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)))
def test_fuzz_sub(constructor1, constructor2, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor1).__sub__(ROOT.Math.Polar2DVector(*constructor2))
    vec1 = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor1))), coordinates
    )()
    vec2 = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor2))), coordinates
    )()
    res_vec = vec1.subtract(vec2)
    assert (ref_vec.R() == pytest.approx(res_vec.rho)  and
            ref_vec.Phi() == pytest.approx(res_vec.phi))


# Run a test that compares ROOT's '__neg__' with vector's '__neg__' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_neg(constructor, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor).__neg__()
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )().__neg__
    assert (ref_vec.R() == pytest.approx(vec().rho)  and
            ref_vec.Phi() == pytest.approx(vec().phi))


# Run the same tests within hypothesis
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                             st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)))
def test_fuzz_neg(constructor, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor).__neg__()
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )().__neg__
    assert (ref_vec.R() == pytest.approx(vec().rho)  and
            ref_vec.Phi() == pytest.approx(vec().phi))


# Run a test that compares ROOT's '__mul__' with vector's 'mul' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_mul(constructor, scalar, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor).__mul__(scalar)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )().__mul__(scalar)
    assert (ref_vec.R() == pytest.approx(vec.rho)  and
            ref_vec.Phi() == pytest.approx(vec.phi))


# Run the same tests within hypothesis
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                             st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)),
       scalar=st.floats(min_value=-10e7, max_value=10e7)
            |  st.integers(min_value=-10e7, max_value=10e7))
def test_fuzz_mul(constructor, scalar, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor).__mul__(scalar)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )().__mul__(scalar)
    assert (ref_vec.R() == pytest.approx(vec.rho)  and
            ref_vec.Phi() == pytest.approx(vec.phi))


# Run a test that compares ROOT's '__truediv__' with vector's '__truediv__' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_truediv(constructor, scalar, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor).__truediv__(scalar)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )().__truediv__(scalar)
    assert (ref_vec.R() == pytest.approx(vec.rho) and
            ref_vec.Phi() == pytest.approx(vec.phi))


# Run the same tests within hypothesis
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                             st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)),
       scalar=st.floats(min_value=-10e7, max_value=10e7)
            |  st.integers(min_value=-10e7, max_value=10e7))
def test_fuzz_truediv(constructor, scalar, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor).__truediv__(scalar)
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )().__truediv__(scalar)
    assert (ref_vec.R() == pytest.approx(vec.rho) and
            ref_vec.Phi() == pytest.approx(vec.phi))


# Run a test that compares ROOT's '__eq__' with vector's 'isclose' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_eq(constructor, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor).__eq__(ROOT.Math.Polar2DVector(*constructor))
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )().isclose(getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )()
    )
    assert ref_vec == vec


# Run the same tests within hypothesis
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7),
                             st.floats(min_value=-10e7, max_value=10e7))
                  | st.tuples(st.integers(min_value=-10e7, max_value=10e7),
                              st.integers(min_value=-10e7, max_value=10e7)))
def test_fuzz_eq(constructor, coordinates):
    ref_vec = ROOT.Math.Polar2DVector(*constructor).__eq__(ROOT.Math.Polar2DVector(*constructor))
    vec = getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )().equal(getattr(
        vector.obj(**dict(zip(["rho", "phi"], constructor))), coordinates
    )()
    )
    assert ref_vec == vec
