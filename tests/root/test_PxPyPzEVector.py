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
    (0, 0, 1, 0), # theta == 0.0
    (0, 0, -1, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 4294967296),
    (0, 4294967296, 0, 0),
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
# Dot
@pytest.mark.parametrize("constructor", constructor)
def test_Dot(constructor, coordinates):
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
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7)) | st.tuples(st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7)))
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
@given(constructor=st.tuples(st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7), st.floats(min_value=-10e7, max_value=10e7)) | st.tuples(st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7), st.integers(min_value=-10e7, max_value=10e7)))
def test_fuzz_M(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).M() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().tau
    )


# Run a test that compares ROOT's 'Mt2()' with vector's 'mt2' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Mt2(constructor, coordinates):
    v = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )()
    assert ROOT.Math.PxPyPzEVector(*constructor).Mt2() == pytest.approx(
        v.t*v.t - v.z*v.z
    )

# Run a test that compares ROOT's 'Mt()' with vector's 'mt' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Mt(constructor, coordinates):
    v = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )()
    assert ROOT.Math.PxPyPzEVector(*constructor).mt() == pytest.approx(
        v.lib.sqrt(v.t*v.t - v.z*v.z)
    )

# Run a test that compares ROOT's 'P2()' with vector's 'mag2' for all cases.
# P2 is our mag2 (ROOT's 4D mag2 is the dot product with itself, what we call tau or mass)
@pytest.mark.parametrize("constructor", constructor)
def test_P2(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).P2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().mag2
    )

# Run a test that compares ROOT's 'P()' with vector's 'mag' for all cases.
# P is our mag (same deal)
@pytest.mark.parametrize("constructor", constructor)
def test_P(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).P() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().mag
    )

# Run a test that compares ROOT's 'Perp2()' with vector's 'rho2' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Perp2(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).Perp2() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().rho2
    )

# Run a test that compares ROOT's 'Perp()' with vector's 'rho' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Perp(constructor, coordinates):
    assert ROOT.Math.sqrt(ROOT.Math.PxPyPzEVector(*constructor).Perp2()) == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().rho
    )

# Run a test that compares ROOT's 'Phi()' with vector's 'phi' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Phi(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).Phi() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().phi
    )

# Run a test that compares ROOT's 'Rapidity()' with vector's 'rapidity' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Rapidity(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).Rapidity() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().rapidity
    )

# Run a test that compares ROOT's 'Beta()' with vector's 'beta' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Beta(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).Beta() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().beta
    )

# Run a test that compares ROOT's 'Eta()' with vector's 'eta' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Eta(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).Eta() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().eta
    )

# Run a test that compares ROOT's 'Gamma()' with vector's 'gamma' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Gamma(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).Gamma() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().gamma
    )

# Run a test that compares ROOT's 'isLightlike()' with vector's 'is_lightlike' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_isLightlike(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).isLightlike() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().is_lightlike()
    )

# Run a test that compares ROOT's 'isSpacelike()' with vector's 'is_spacelike' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_isSpacelike(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).isSpacelike() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().is_spacelike()
    )

# Run a test that compares ROOT's 'isTimelike()' with vector's 'is_timelike' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_isTimelike(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).isTimelike() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().is_timelike()
    )

# Run a test that compares ROOT's 'Theta()' with vector's 'theta' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Theta(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).Theta() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().theta
    )

# Run a test that compares ROOT's 'X()' with vector's 'x' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_X(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).X() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().x
    )

# Run a test that compares ROOT's 'Y()' with vector's 'y' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Y(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).Y() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().y
    )

# Run a test that compares ROOT's 'Z()' with vector's 'z' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_Z(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).Z() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().z
    )

# Run a test that compares ROOT's 'T()' with vector's 't' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_T(constructor, coordinates):
    assert ROOT.Math.PxPyPzEVector(*constructor).T() == pytest.approx(
        getattr(
            vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
        )().t
    )


# Run a test that compares ROOT's '__add__' with vector's 'add' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_add(constructor, coordinates):
    ref_vec = ROOT.Math.PxPyPzEVector(*constructor).__add__(ROOT.Math.PxPyPzEVector(*constructor))
    vec = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )().add(getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates)())
    assert ref_vec.X() == pytest.approx(vec.x)
    assert ref_vec.Y() == pytest.approx(vec.y)
    assert ref_vec.Z() == pytest.approx(vec.z)
    assert ref_vec.T() == pytest.approx(vec.t)


# Run a test that compares ROOT's '__sub__' with vector's 'subtract' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_sub(constructor, coordinates):
    ref_vec = ROOT.Math.PxPyPzEVector(*constructor).__sub__(ROOT.Math.PxPyPzEVector(*constructor))
    vec1 = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )()
    vec2 = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )()
    res_vec = vec1.subtract(vec2)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)
    assert ref_vec.T() == pytest.approx(res_vec.t)

# Run a test that compares ROOT's '__neg__' with vector's '__neg__' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_neg(constructor, coordinates):
    ref_vec = ROOT.Math.PxPyPzEVector(*constructor).__neg__()
    vec = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )().__neg__
    assert ref_vec.X() == pytest.approx(vec().x)
    assert ref_vec.Y() == pytest.approx(vec().y)
    assert ref_vec.Z() == pytest.approx(vec().z)
    assert ref_vec.T() == pytest.approx(vec().t)


# Run a test that compares ROOT's '__mul__' with vector's 'mul' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_mul(constructor, scalar, coordinates):
    ref_vec = ROOT.Math.PxPyPzEVector(*constructor).__mul__(scalar)
    vec = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )().__mul__(scalar)
    assert ref_vec.X() == pytest.approx(vec.x)
    assert ref_vec.Y() == pytest.approx(vec.y)
    assert ref_vec.Z() == pytest.approx(vec.z)
    assert ref_vec.T() == pytest.approx(vec.t)


# Run a test that compares ROOT's '__truediv__' with vector's '__truediv__' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_truediv(constructor, scalar, coordinates):
    ref_vec = ROOT.Math.PxPyPzEVector(*constructor).__truediv__(scalar)
    vec = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )().__truediv__(scalar)
    assert ref_vec.X() == pytest.approx(vec.x)
    assert ref_vec.Y() == pytest.approx(vec.y)
    assert ref_vec.Z() == pytest.approx(vec.z)
    assert ref_vec.T() == pytest.approx(vec.t)


# Run a test that compares ROOT's '__eq__' with vector's 'isclose' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_eq(constructor, coordinates):
    ref_vec = ROOT.Math.PxPyPzEVector(*constructor).__eq__(ROOT.Math.PxPyPzEVector(*constructor))
    vec = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )().isclose(getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )()
    )
    assert ref_vec == vec


# Run a test that compares ROOT's 'RotateX()' with vector's 'rotateX' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_RotateX(constructor, angle, coordinates):
    ref_vec = ROOT.Math.RotationX(angle)*ROOT.Math.PxPyPzEVector(*constructor)
    vec = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )()
    res_vec = vec.rotateX(angle)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)
    assert ref_vec.T() == pytest.approx(res_vec.t)


# Run a test that compares ROOT's 'RotateY()' with vector's 'rotateY' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_RotateY(constructor, angle, coordinates):
    ref_vec = ROOT.Math.RotationY(angle)*ROOT.Math.PxPyPzEVector(*constructor)
    vec = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )()
    res_vec = vec.rotateY(angle)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)
    assert ref_vec.T() == pytest.approx(res_vec.t)


# Run a test that compares ROOT's 'RotateZ()' with vector's 'rotateZ' for all cases.
@pytest.mark.parametrize("constructor", constructor)
def test_RotateZ(constructor, angle, coordinates):
    ref_vec = ROOT.Math.RotationZ(angle)*ROOT.Math.PxPyPzEVector(*constructor)
    vec = getattr(
        vector.obj(**dict(zip(["x", "y", "z", "t"], constructor))), coordinates
    )()
    res_vec = vec.rotateZ(angle)
    assert ref_vec.X() == pytest.approx(res_vec.x)
    assert ref_vec.Y() == pytest.approx(res_vec.y)
    assert ref_vec.Z() == pytest.approx(res_vec.z)
    assert ref_vec.T() == pytest.approx(res_vec.t)
