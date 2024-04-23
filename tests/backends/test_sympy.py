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
M, m, mass, E, e, energy = sympy.symbols("M m mass E e energy")


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
        with pytest.raises(TypeError):
            vec = vec_cls(bad=1, **coords)
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
        with pytest.raises(TypeError):
            vec = vec_cls(bad=1, **coords)

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
        with pytest.raises(TypeError):
            vec = vec_cls(bad=1, **coords)

        assert vec.z == z
        assert vec.eta == sympy.asinh(z / sympy.sqrt(x**2 + y**2))
        assert vec.theta == sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))

        assert isinstance(vec.longitudinal, vector.backends.sympy.LongitudinalSympyZ)
        assert vec.longitudinal.elements == (z,)

        coords = {"x": x, "y": y, "eta": eta}
        if vec_cls == vector.VectorSympy4D:
            coords["t"] = t
        vec = vec_cls(**coords)
        with pytest.raises(TypeError):
            vec = vec_cls(bad=1, **coords)

        assert vec.z == sympy.sqrt(x**2 + y**2) * sympy.sinh(eta)
        assert vec.eta == eta
        assert vec.theta == 2.0 * sympy.atan(sympy.exp(-eta))

        assert isinstance(vec.longitudinal, vector.backends.sympy.LongitudinalSympyEta)
        assert vec.longitudinal.elements == (eta,)

        coords = {"x": x, "y": y, "theta": theta}
        if vec_cls == vector.VectorSympy4D:
            coords["t"] = t
        vec = vec_cls(**coords)
        with pytest.raises(TypeError):
            vec = vec_cls(bad=1, **coords)

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
        with pytest.raises(TypeError):
            vec = vec_cls(bad=1, **coords)

        assert vec.t == t
        assert vec.tau == sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))

        assert isinstance(vec.temporal, vector.backends.sympy.TemporalSympyT)
        assert vec.temporal.elements == (t,)

        coords = {"x": x, "y": y, "z": z, "tau": tau}
        vec = vec_cls(**coords)
        with pytest.raises(TypeError):
            vec = vec_cls(bad=1, **coords)

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
        with pytest.raises(TypeError):
            vec = vec_cls(bad=1, **coords)

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
        with pytest.raises(TypeError):
            vec = vec_cls(bad=1, **coords)

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
        with pytest.raises(TypeError):
            vec = vec_cls(bad=1, **coords)

        assert vec.pz == pz
        assert vec.eta == sympy.asinh(pz / sympy.sqrt(px**2 + py**2))
        assert vec.theta == sympy.acos(pz / sympy.sqrt(px**2 + py**2 + pz**2))

        assert isinstance(vec.longitudinal, vector.backends.sympy.LongitudinalSympyZ)
        assert vec.longitudinal.elements == (pz,)

    # 4D coords
    for vec_cls in (vector.MomentumSympy4D,):
        coords = {"px": px, "py": py, "pz": pz, "E": E}
        vec = vec_cls(**coords)
        with pytest.raises(TypeError):
            vec = vec_cls(bad=1, **coords)

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
        with pytest.raises(TypeError):
            vec = vec_cls(bad=1, **coords)

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
        vector.VectorSympy2D()

    with pytest.raises(TypeError):
        vector.VectorSympy3D(x=1, y=2, z=3)

    with pytest.raises(TypeError):
        vector.VectorSympy3D()

    with pytest.raises(TypeError):
        vector.VectorSympy4D(x=1, y=2, z=3, t=4)

    with pytest.raises(TypeError):
        vector.VectorSympy4D()


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

    v2 = vector.VectorSympy3D(x=x, y=y, z=z)
    assert abs(v2) == sympy.sqrt(x**2 + y**2 + z**2)

    v3 = vector.VectorSympy4D(x=x, y=y, z=z, t=t)
    assert abs(v3) == sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))


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


def test_reprs():
    assert (
        vector.backends.sympy.AzimuthalSympyXY(x, y).__repr__()
        == "AzimuthalSympyXY(x=x, y=y)"
    )
    assert (
        vector.backends.sympy.AzimuthalSympyRhoPhi(rho, phi).__repr__()
        == "AzimuthalSympyRhoPhi(rho=rho, phi=phi)"
    )
    assert (
        vector.backends.sympy.LongitudinalSympyZ(z).__repr__()
        == "LongitudinalSympyZ(z=z)"
    )
    assert (
        vector.backends.sympy.LongitudinalSympyEta(eta).__repr__()
        == "LongitudinalSympyEta(eta=eta)"
    )
    assert (
        vector.backends.sympy.LongitudinalSympyTheta(theta).__repr__()
        == "LongitudinalSympyTheta(theta=theta)"
    )
    assert vector.backends.sympy.TemporalSympyT(t).__repr__() == "TemporalSympyT(t=t)"
    assert (
        vector.backends.sympy.TemporalSympyTau(tau).__repr__()
        == "TemporalSympyTau(tau=tau)"
    )

    assert vector.VectorSympy2D(x=x, y=y).__repr__() == "VectorSympy2D(x=x, y=y)"
    assert (
        vector.VectorSympy3D(x=x, y=y, z=z).__repr__() == "VectorSympy3D(x=x, y=y, z=z)"
    )
    assert (
        vector.VectorSympy4D(x=x, y=y, z=z, t=t).__repr__()
        == "VectorSympy4D(x=x, y=y, z=z, t=t)"
    )
    assert (
        vector.MomentumSympy2D(px=px, py=py).__repr__()
        == "MomentumSympy2D(px=px, py=py)"
    )
    assert (
        vector.MomentumSympy3D(px=px, py=py, pz=pz).__repr__()
        == "MomentumSympy3D(px=px, py=py, pz=pz)"
    )
    assert (
        vector.MomentumSympy4D(px=px, py=py, pz=pz, M=M).__repr__()
        == "MomentumSympy4D(px=px, py=py, pz=pz, mass=M)"
    )


def test_setters():
    v = vector.VectorSympy2D(x=x, y=y)
    v.x = 2 * x
    assert v.x == 2 * x
    v.y = 2 * y
    assert v.y == 2 * y
    v.rho = 2 * rho
    assert v.rho == 2 * rho
    v.phi = 2 * phi
    assert v.phi == 2 * phi

    v = vector.MomentumSympy2D(px=px, py=py)
    v.px = 2 * px
    assert v.px == 2 * px
    v.py = 2 * py
    assert v.py == 2 * py
    v.pt = 2 * pt
    assert v.pt == 2 * pt

    v = vector.VectorSympy3D(x=x, y=y, z=z)
    v.x = 2 * x
    assert v.x == 2 * x
    v.y = 2 * y
    assert v.y == 2 * y
    v.z = 2 * z
    assert v.z == 2 * z
    v.eta = 2 * eta
    assert v.eta == 2 * eta
    v.theta = 2 * theta
    assert v.theta == 2 * theta

    v = vector.MomentumSympy3D(px=px, py=py, pz=pz)
    v.px = 2 * px
    assert v.px == 2 * px
    v.py = 2 * py
    assert v.py == 2 * py
    v.pt = 2 * pt
    assert v.pt == 2 * pt
    v.pz = 2 * pz
    assert v.pz == 2 * pz

    v = vector.VectorSympy4D(x=x, y=y, z=z, t=t)
    v.x = 2 * x
    assert v.x == 2 * x
    v.y = 2 * y
    assert v.y == 2 * y
    v.z = 2 * z
    assert v.z == 2 * z
    v.eta = 2 * eta
    assert v.eta == 2 * eta
    v.theta = 2 * theta
    assert v.theta == 2 * theta
    v.t = 2 * t
    assert v.t == 2 * t
    v.tau = 2 * tau
    assert v.tau == 2 * tau

    v = vector.MomentumSympy4D(px=px, py=py, pz=pz, m=M)
    v.px = 2 * px
    assert v.px == 2 * px
    v.py = 2 * py
    assert v.py == 2 * py
    v.pt = 2 * pt
    assert v.pt == 2 * pt
    v.pz = 2 * pz
    assert v.pz == 2 * pz
    v.m = 2 * m
    assert v.m == 2 * m
    v.mass = 2 * mass
    assert v.mass == 2 * mass
    v.M = 2 * M
    assert v.M == 2 * M
    v.e = 2 * e
    assert v.e == 2 * e
    v.energy = 2 * energy
    assert v.energy == 2 * energy
    v.E = 2 * E
    assert v.E == 2 * E
