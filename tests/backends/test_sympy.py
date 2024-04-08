# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y = sympy.symbols("x y")
rho, phi = sympy.symbols("rho phi")
z, eta, theta = sympy.symbols("z eta theta")
t, tau = sympy.symbols("t tau")
px, py = sympy.symbols("px py")
pt = sympy.symbols("pt")
pz = sympy.symbols("pz")
M, E = sympy.symbols("M E")


def test_construction():
    # generic
    # 2D coords
    for vec_cls in (vector.VectorSympy2D, vector.VectorSympy3D, vector.VectorSympy4D):
        coords = {"x": x, "y": y}
        if vec_cls in (vector.VectorSympy3D, vector.VectorSympy4D):
            coords["z"] = z
        if vec_cls == vector.VectorSympy4D:
            coords["t"] = t
        vec = vec_cls(**coords)
        assert vec.x == x
        assert vec.y == y
        assert vec.phi == sympy.atan2(y, x)
        assert vec.rho == sympy.sqrt(x**2 + y**2)

        assert isinstance(vec.azimuthal, vector.backends.sympy.AzimuthalSympyXY)
        assert vec.azimuthal.elements == (x, y)

        coords = {"rho": rho, "phi": phi}
        if vec_cls in (vector.VectorSympy3D, vector.VectorSympy4D):
            coords["z"] = z
        if vec_cls == vector.VectorSympy4D:
            coords["t"] = t
        vec = vec_cls(**coords)
        assert vec.x == rho * sympy.cos(phi)
        assert vec.y == rho * sympy.sin(phi)
        assert vec.phi == phi
        assert vec.rho == rho

        assert isinstance(vec.azimuthal, vector.backends.sympy.AzimuthalSympyRhoPhi)
        assert vec.azimuthal.elements == (rho, phi)

    # 3D coords
    for vec_cls in (vector.VectorSympy3D, vector.VectorSympy4D):
        coords = {"x": x, "y": y, "z": z}
        if vec_cls == vector.VectorSympy4D:
            coords["t"] = t
        vec = vec_cls(**coords)
        assert vec.z == z
        assert vec.eta == sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        assert vec.theta == sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))

        assert isinstance(vec.longitudinal, vector.backends.sympy.LongitudinalSympyZ)
        assert vec.longitudinal.elements == (z,)

        coords = {"x": x, "y": y, "eta": eta}
        if vec_cls == vector.VectorSympy4D:
            coords["t"] = t
        vec = vec_cls(**coords)
        assert vec.z == sympy.sqrt(x**2 + y**2) * sympy.sinh(eta)
        assert vec.eta == eta
        assert vec.theta == 2.0 * sympy.atan(sympy.exp(-eta))

        assert isinstance(vec.longitudinal, vector.backends.sympy.LongitudinalSympyEta)
        assert vec.longitudinal.elements == (eta,)

        coords = {"x": x, "y": y, "theta": theta}
        if vec_cls == vector.VectorSympy4D:
            coords["t"] = t
        vec = vec_cls(**coords)
        assert vec.z == sympy.sqrt(x**2 + y**2) / sympy.tan(theta)
        assert vec.eta == -sympy.log(sympy.tan(0.5 * theta))
        assert vec.theta == theta

        assert isinstance(
            vec.longitudinal, vector.backends.sympy.LongitudinalSympyTheta
        )
        assert vec.longitudinal.elements == (theta,)

    # 4D coords
    for vec_cls in (vector.VectorSympy4D,):
        coords = {"x": x, "y": y, "z": z, "t": t}
        vec = vec_cls(**coords)
        assert vec.t == t
        assert vec.tau == sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))

        assert isinstance(vec.temporal, vector.backends.sympy.TemporalSympyT)
        assert vec.temporal.elements == (t,)

        coords = {"x": x, "y": y, "z": z, "tau": tau}
        vec = vec_cls(**coords)
        assert vec.t == sympy.sqrt(tau**2 + x**2 + y**2 + z**2)
        assert vec.tau == tau

        assert isinstance(vec.temporal, vector.backends.sympy.TemporalSympyTau)
        assert vec.temporal.elements == (tau,)

    # momentum
    # 2D coords
    for vec_cls in (
        vector.MomentumSympy2D,
        vector.MomentumSympy3D,
        vector.MomentumSympy4D,
    ):
        coords = {"px": px, "py": py}
        if vec_cls in (vector.MomentumSympy3D, vector.MomentumSympy4D):
            coords["z"] = z
        if vec_cls == vector.MomentumSympy4D:
            coords["t"] = t
        vec = vec_cls(**coords)
        assert vec.px == px
        assert vec.py == py
        assert vec.phi == sympy.atan2(py, px)
        assert vec.rho == sympy.sqrt(px**2 + py**2)
        assert vec.pt == sympy.sqrt(px**2 + py**2)

        assert isinstance(vec.azimuthal, vector.backends.sympy.AzimuthalSympyXY)
        assert vec.azimuthal.elements == (px, py)

        coords = {"pt": pt, "phi": phi}
        if vec_cls in (vector.MomentumSympy3D, vector.MomentumSympy4D):
            coords["z"] = z
        if vec_cls == vector.MomentumSympy4D:
            coords["t"] = t
        vec = vec_cls(**coords)
        assert vec.px == pt * sympy.cos(phi)
        assert vec.py == pt * sympy.sin(phi)
        assert vec.phi == phi
        assert vec.rho == pt
        assert vec.pt == pt

        assert isinstance(vec.azimuthal, vector.backends.sympy.AzimuthalSympyRhoPhi)
        assert vec.azimuthal.elements == (pt, phi)

    # 3D coords
    for vec_cls in (vector.MomentumSympy3D, vector.MomentumSympy4D):
        coords = {"px": px, "py": py, "pz": pz}
        if vec_cls == vector.MomentumSympy4D:
            coords["t"] = t
        vec = vec_cls(**coords)
        assert vec.pz == pz
        assert vec.eta == sympy.asinh(pz / sympy.sqrt(px**2 + py**2))
        assert vec.theta == sympy.acos(pz / sympy.sqrt(px**2 + py**2 + pz**2))

        assert isinstance(vec.longitudinal, vector.backends.sympy.LongitudinalSympyZ)
        assert vec.longitudinal.elements == (pz,)

    # 4D coords
    for vec_cls in (vector.MomentumSympy4D,):
        coords = {"px": px, "py": py, "pz": pz, "E": E}
        vec = vec_cls(**coords)
        assert vec.m == sympy.sqrt(sympy.Abs(px**2 + py**2 + pz**2 - E**2))
        assert sympy.sqrt(sympy.Abs(px**2 + py**2 + pz**2 - E**2)) == vec.M
        assert vec.mass == sympy.sqrt(sympy.Abs(px**2 + py**2 + pz**2 - E**2))
        assert vec.e == E
        assert vec.E == E
        assert vec.energy == E

        assert isinstance(vec.temporal, vector.backends.sympy.TemporalSympyT)
        assert vec.temporal.elements == (E,)

        coords = {"px": px, "py": py, "pz": pz, "M": M}
        vec = vec_cls(**coords)
        assert vec.m == M
        assert vec.M == M
        assert vec.mass == M
        assert vec.e == sympy.sqrt(M**2 + px**2 + py**2 + pz**2)
        assert sympy.sqrt(M**2 + px**2 + py**2 + pz**2) == vec.E
        assert vec.energy == sympy.sqrt(M**2 + px**2 + py**2 + pz**2)

        assert isinstance(vec.temporal, vector.backends.sympy.TemporalSympyTau)
        assert vec.temporal.elements == (M,)


def test_type_checks():
    with pytest.raises(TypeError):
        vector.VectorSympy2D(x=1, y=2)

    with pytest.raises(TypeError):
        vector.VectorSympy3D(x=1, y=2, z=3)

    with pytest.raises(TypeError):
        vector.VectorSympy4D(x=1, y=2, z=3, t=4)


nx, ny, nz = sympy.symbols("nx ny nz")


def test_eq():
    v1 = vector.VectorSympy2D(x=x, y=y)
    v2 = vector.VectorSympy2D(x=nx, y=ny)
    assert v1 == v1  # noqa: PLR0124
    assert not v1 == v2  # noqa: SIM201
    with pytest.raises(TypeError):
        v1.equal(v2.to_Vector3D())


def test_ne():
    v1 = vector.VectorSympy2D(x=x, y=y)
    v2 = vector.VectorSympy2D(x=nx, y=ny)
    assert not v1 != v1  # noqa: PLR0124,SIM202
    assert v1 != v2
    with pytest.raises(TypeError):
        v1.not_equal(v2.to_Vector3D())


def test_abs():
    v1 = vector.VectorSympy2D(x=x, y=y)
    assert abs(v1) == sympy.sqrt(x**2 + y**2)


def test_add():
    v1 = vector.VectorSympy2D(x=x, y=y)
    v2 = vector.VectorSympy2D(x=nx, y=ny)
    assert v1 + v2 == vector.VectorSympy2D(x=x + nx, y=y + ny)
    assert v1 + v2.to_Vector3D().like(v1) == vector.VectorSympy2D(x=x + nx, y=y + ny)
    v1 += v2
    assert v1 == vector.VectorSympy2D(x=x + nx, y=y + ny)
    with pytest.raises(TypeError):
        v1 + 5
    with pytest.raises(TypeError):
        5 + v1
    with pytest.raises(TypeError):
        v1 + v2.to_Vector3D()


def test_sub():
    v1 = vector.VectorSympy2D(x=x, y=y)
    v2 = vector.VectorSympy2D(x=nx, y=ny)
    assert v1 - v2 == vector.VectorSympy2D(x=x - nx, y=y - ny)
    v1 -= v2
    assert v1 == vector.VectorSympy2D(x=x - nx, y=y - ny)
    with pytest.raises(TypeError):
        v1 - 5
    with pytest.raises(TypeError):
        5 - v1
    with pytest.raises(TypeError):
        v1 - v2.to_Vector3D()


def test_mul():
    v1 = vector.VectorSympy2D(x=x, y=y)
    v2 = vector.VectorSympy2D(x=nx, y=ny)
    assert v1 * 10 == vector.VectorSympy2D(x=x * 10, y=y * 10)
    assert 10 * v1 == vector.VectorSympy2D(x=10 * x, y=10 * y)
    v1 *= 10
    assert v1 == vector.VectorSympy2D(x=10 * x, y=10 * y)
    with pytest.raises(TypeError):
        v1 * v2


def test_neg():
    v1 = vector.VectorSympy2D(x=x, y=y)
    assert -v1 == vector.VectorSympy2D(x=-x, y=-y)


def test_pos():
    v1 = vector.VectorSympy2D(x=x, y=y)
    assert +v1 == vector.VectorSympy2D(x=x, y=y)


def test_truediv():
    v1 = vector.VectorSympy2D(x=x, y=y)
    v2 = vector.VectorSympy2D(x=nx, y=ny)
    assert v1 / 10 == vector.VectorSympy2D(x=0.1 * x, y=0.1 * y)
    v1 /= 10
    assert v1 == vector.VectorSympy2D(x=0.1 * x, y=0.1 * y)
    with pytest.raises(TypeError):
        10 / v1
    with pytest.raises(TypeError):
        v1 / v2


def test_pow():
    v1 = vector.VectorSympy2D(x=x, y=y)
    v2 = vector.VectorSympy2D(x=nx, y=ny)
    assert v1**2 == x**2 + y**2
    with pytest.raises(TypeError):
        2**v1
    with pytest.raises(TypeError):
        v1**v2


def test_matmul():
    v1 = vector.VectorSympy2D(x=x, y=y)
    v2 = vector.VectorSympy2D(x=nx, y=ny)
    assert v1 @ v2 == x * nx + y * ny
    assert v2 @ v1 == x * nx + y * ny
    with pytest.raises(TypeError):
        v1 @ 5
