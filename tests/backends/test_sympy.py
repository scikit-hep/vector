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


def test_conversion_2D():
    v = vector.VectorSympy2D(x=x, y=y)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.VectorSympy2D)
    assert tv.x == x
    assert tv.y == y
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.VectorSympy3D)
    assert tv.x == x
    assert tv.y == y
    assert tv.z == 0.0
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.VectorSympy4D)
    assert tv.x == x
    assert tv.y == y
    assert tv.z == 0.0
    assert tv.t == 0.0

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.VectorSympy2D)
        if azimuthal == "xy":
            assert tv.x == x
            assert tv.y == y
        elif azimuthal == "rhophi":
            assert tv.rho == sympy.sqrt(x**2 + y**2)
            assert tv.phi == sympy.atan2(y, x)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.VectorSympy3D)
            assert tv.x == x
            assert tv.y == y
            assert getattr(tv, longitudinal) == 0.0

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.VectorSympy4D)
                assert tv.x == x
                assert tv.y == y
                assert getattr(tv, longitudinal) == 0.0
                assert getattr(tv, temporal) == 0.0


def test_momentum_conversion_2D():
    v = vector.MomentumSympy2D(px=px, py=py)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.MomentumSympy2D)
    assert tv.x == px
    assert tv.y == py
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.MomentumSympy3D)
    assert tv.x == px
    assert tv.y == py
    assert tv.z == 0.0
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.MomentumSympy4D)
    assert tv.x == px
    assert tv.y == py
    assert tv.z == 0.0
    assert tv.t == 0.0

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.MomentumSympy2D)
        if azimuthal == "xy":
            assert tv.x == px
            assert tv.y == py
        elif azimuthal == "rhophi":
            assert tv.rho == sympy.sqrt(px**2 + py**2)
            assert tv.phi == sympy.atan2(py, px)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.MomentumSympy3D)
            assert tv.x == px
            assert tv.y == py
            assert getattr(tv, longitudinal) == 0.0

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.MomentumSympy4D)
                assert tv.x == px
                assert tv.y == py
                assert getattr(tv, longitudinal) == 0.0
                assert getattr(tv, temporal) == 0.0


def test_conversion_3D():
    v = vector.VectorSympy3D(x=x, y=y, z=z)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.VectorSympy2D)
    assert tv.x == x
    assert tv.y == y
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.VectorSympy3D)
    assert tv.x == x
    assert tv.y == y
    assert tv.z == z
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.VectorSympy4D)
    assert tv.x == x
    assert tv.y == y
    assert tv.z == z
    assert tv.t == 0.0

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.VectorSympy2D)
        if azimuthal == "xy":
            assert tv.x == x
            assert tv.y == y
        elif azimuthal == "rhophi":
            assert tv.rho == sympy.sqrt(x**2 + y**2)
            assert tv.phi == sympy.atan2(y, x)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.VectorSympy3D)
            if azimuthal == "xy":
                assert tv.x == x
                assert tv.y == y
            elif azimuthal == "rhophi":
                assert tv.rho == sympy.sqrt(x**2 + y**2)
                assert tv.phi == sympy.atan2(y, x)

            if longitudinal == "z":
                assert tv.z == z
            elif longitudinal == "eta":
                assert tv.eta == sympy.asinh(z / sympy.sqrt(x**2 + y**2))
            elif longitudinal == "theta":
                assert tv.theta == sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.VectorSympy4D)
                if azimuthal == "xy":
                    assert tv.x == x
                    assert tv.y == y
                elif azimuthal == "rhophi":
                    assert tv.rho == sympy.sqrt(x**2 + y**2)
                    assert tv.phi == sympy.atan2(y, x)

                if longitudinal == "z":
                    assert tv.z == z
                elif longitudinal == "eta":
                    assert tv.eta == sympy.asinh(z / sympy.sqrt(x**2 + y**2))
                elif longitudinal == "theta":
                    assert tv.theta == sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))
                assert getattr(tv, temporal) == pytest.approx(0)


def test_momentum_conversion_3D():
    v = vector.MomentumSympy3D(px=px, py=py, pz=pz)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.MomentumSympy2D)
    assert tv.x == px
    assert tv.y == py
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.MomentumSympy3D)
    assert tv.x == px
    assert tv.y == py
    assert tv.z == pz
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.MomentumSympy4D)
    assert tv.x == px
    assert tv.y == py
    assert tv.z == pz
    assert tv.t == 0.0

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.MomentumSympy2D)
        if azimuthal == "xy":
            assert tv.x == px
            assert tv.y == py
        elif azimuthal == "rhophi":
            assert tv.rho == sympy.sqrt(px**2 + py**2)
            assert tv.phi == sympy.atan2(py, px)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.MomentumSympy3D)
            if azimuthal == "xy":
                assert tv.x == px
                assert tv.y == py
            elif azimuthal == "rhophi":
                assert tv.rho == sympy.sqrt(px**2 + py**2)
                assert tv.phi == sympy.atan2(py, px)

            if longitudinal == "z":
                assert tv.z == pz
            elif longitudinal == "eta":
                assert tv.eta == sympy.asinh(pz / sympy.sqrt(px**2 + py**2))
            elif longitudinal == "theta":
                assert tv.theta == sympy.acos(pz / sympy.sqrt(px**2 + py**2 + pz**2))

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.MomentumSympy4D)
                if azimuthal == "xy":
                    assert tv.x == px
                    assert tv.y == py
                elif azimuthal == "rhophi":
                    assert tv.rho == sympy.sqrt(px**2 + py**2)
                    assert tv.phi == sympy.atan2(py, px)

                if longitudinal == "z":
                    assert tv.z == pz
                elif longitudinal == "eta":
                    assert tv.eta == sympy.asinh(pz / sympy.sqrt(px**2 + py**2))
                elif longitudinal == "theta":
                    assert tv.theta == sympy.acos(
                        pz / sympy.sqrt(px**2 + py**2 + pz**2)
                    )
                assert getattr(tv, temporal) == pytest.approx(0)


def test_conversion_4D():
    v = vector.VectorSympy4D(x=x, y=y, z=z, t=t)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.VectorSympy2D)
    assert tv.x == x
    assert tv.y == y
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.VectorSympy3D)
    assert tv.x == x
    assert tv.y == y
    assert tv.z == z
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.VectorSympy4D)
    assert tv.x == x
    assert tv.y == y
    assert tv.z == z
    assert tv.t == t

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.VectorSympy2D)
        if azimuthal == "xy":
            assert tv.x == x
            assert tv.y == y
        elif azimuthal == "rhophi":
            assert tv.rho == sympy.sqrt(x**2 + y**2)
            assert tv.phi == sympy.atan2(y, x)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.VectorSympy3D)
            if azimuthal == "xy":
                assert tv.x == x
                assert tv.y == y
            elif azimuthal == "rhophi":
                assert tv.rho == sympy.sqrt(x**2 + y**2)
                assert tv.phi == sympy.atan2(y, x)

            if longitudinal == "z":
                assert tv.z == z
            elif longitudinal == "eta":
                assert tv.eta == sympy.asinh(z / sympy.sqrt(x**2 + y**2))
            elif longitudinal == "theta":
                assert tv.theta == sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.VectorSympy4D)
                if azimuthal == "xy":
                    assert tv.x == x
                    assert tv.y == y
                elif azimuthal == "rhophi":
                    assert tv.rho == sympy.sqrt(x**2 + y**2)
                    assert tv.phi == sympy.atan2(y, x)

                if longitudinal == "z":
                    assert tv.z == z
                elif longitudinal == "eta":
                    assert tv.eta == sympy.asinh(z / sympy.sqrt(x**2 + y**2))
                elif longitudinal == "theta":
                    assert tv.theta == sympy.acos(z / sympy.sqrt(x**2 + y**2 + z**2))

                if temporal == "t":
                    assert tv.t == t
                elif temporal == "tau":
                    assert tv.tau == sympy.sqrt(sympy.Abs(-(t**2) + x**2 + y**2 + z**2))


def test_momentum_conversion_4D():
    v = vector.MomentumSympy4D(px=px, py=py, pz=pz, E=E)
    tv = v.to_Vector2D()
    assert isinstance(tv, vector.MomentumSympy2D)
    assert tv.x == px
    assert tv.y == py
    tv = v.to_Vector3D()
    assert isinstance(tv, vector.MomentumSympy3D)
    assert tv.x == px
    assert tv.y == py
    assert tv.z == pz
    tv = v.to_Vector4D()
    assert isinstance(tv, vector.MomentumSympy4D)
    assert tv.x == px
    assert tv.y == py
    assert tv.z == pz
    assert tv.t == E

    for azimuthal in "xy", "rhophi":
        tv = getattr(v, "to_" + azimuthal)()
        assert isinstance(tv, vector.MomentumSympy2D)
        if azimuthal == "xy":
            assert tv.x == px
            assert tv.y == py
        elif azimuthal == "rhophi":
            assert tv.rho == sympy.sqrt(px**2 + py**2)
            assert tv.phi == sympy.atan2(py, px)

        for longitudinal in "z", "theta", "eta":
            tv = getattr(v, "to_" + azimuthal + longitudinal)()
            assert isinstance(tv, vector.MomentumSympy3D)
            if azimuthal == "xy":
                assert tv.x == px
                assert tv.y == py
            elif azimuthal == "rhophi":
                assert tv.rho == sympy.sqrt(px**2 + py**2)
                assert tv.phi == sympy.atan2(py, px)

            if longitudinal == "z":
                assert tv.z == pz
            elif longitudinal == "eta":
                assert tv.eta == sympy.asinh(pz / sympy.sqrt(px**2 + py**2))
            elif longitudinal == "theta":
                assert tv.theta == sympy.acos(pz / sympy.sqrt(px**2 + py**2 + pz**2))

            for temporal in "t", "tau":
                tv = getattr(v, "to_" + azimuthal + longitudinal + temporal)()
                assert isinstance(tv, vector.MomentumSympy4D)
                if azimuthal == "xy":
                    assert tv.x == px
                    assert tv.y == py
                elif azimuthal == "rhophi":
                    assert tv.rho == sympy.sqrt(px**2 + py**2)
                    assert tv.phi == sympy.atan2(py, px)

                if longitudinal == "z":
                    assert tv.z == pz
                elif longitudinal == "eta":
                    assert tv.eta == sympy.asinh(pz / sympy.sqrt(px**2 + py**2))
                elif longitudinal == "theta":
                    assert tv.theta == sympy.acos(
                        pz / sympy.sqrt(px**2 + py**2 + pz**2)
                    )

                if temporal == "t":
                    assert tv.t == E
                elif temporal == "tau":
                    assert tv.tau == sympy.sqrt(
                        sympy.Abs(-(E**2) + px**2 + py**2 + pz**2)
                    )


def test_conversion_with_coords():
    # 2D -> 3D
    vec = vector.VectorSympy2D(x=x, y=y)
    assert vec.to_Vector3D(z=z).z == z
    assert vec.to_Vector3D(eta=eta).eta == eta
    assert vec.to_Vector3D(theta=theta).theta == theta

    # test alias
    assert vec.to_3D(z=z).x == vec.x
    assert vec.to_3D(z=z).y == vec.y

    # 2D -> 4D
    assert vec.to_Vector4D(z=z, t=t).z == z
    assert vec.to_Vector4D(z=z, t=t).t == t
    assert vec.to_Vector4D(eta=eta, t=t).eta == eta
    assert vec.to_Vector4D(eta=eta, t=t).t == t
    assert vec.to_Vector4D(theta=theta, t=t).theta == theta
    assert vec.to_Vector4D(theta=theta, t=t).t == t
    assert vec.to_Vector4D(z=z, tau=tau).z == z
    assert vec.to_Vector4D(z=z, tau=tau).tau == tau
    assert vec.to_Vector4D(eta=eta, tau=tau).eta == eta
    assert vec.to_Vector4D(eta=eta, tau=tau).tau == tau
    assert vec.to_Vector4D(theta=theta, tau=tau).theta == theta
    assert vec.to_Vector4D(theta=theta, tau=tau).tau == tau

    # test alias
    assert vec.to_4D(z=z, t=t).x == vec.x
    assert vec.to_4D(z=z, t=t).y == vec.y

    # 3D -> 4D
    vec = vector.VectorSympy3D(x=px, y=py, z=pz)

    # test alias
    assert vec.to_4D(t=t).t == t
    assert vec.to_4D(tau=tau).tau == tau

    assert vec.to_Vector4D(t=t).x == vec.x
    assert vec.to_Vector4D(t=t).y == vec.y
    assert vec.to_Vector4D(t=t).z == vec.z

    # check if momentum coords work
    vec = vector.MomentumSympy2D(px=px, py=py)
    assert vec.to_Vector3D(pz=pz).pz == pz

    # test both alias and original methods
    assert vec.to_4D(pz=pz, m=M).pz == pz
    assert vec.to_4D(pz=pz, m=M).m == M
    assert vec.to_4D(pz=pz, mass=M).mass == M
    assert vec.to_4D(pz=pz, M=M).M == M
    assert vec.to_Vector4D(pz=pz, e=E).e == E
    assert vec.to_Vector4D(pz=pz, energy=E).energy == E
    assert vec.to_Vector4D(pz=pz, E=E).E == E

    vec = vector.MomentumSympy3D(px=px, py=py, pz=pz)

    # test both alias and original methods
    assert vec.to_4D(m=M).m == M
    assert vec.to_4D(mass=M).mass == M
    assert vec.to_4D(M=M).M == M
    assert vec.to_Vector4D(e=E).e == E
    assert vec.to_Vector4D(energy=E).energy == E
    assert vec.to_Vector4D(E=E).E == E


def test_like():
    v1 = vector.VectorSympy2D(x=x, y=y)
    v2 = vector.VectorSympy3D(x=x, y=y, z=z)
    v3 = vector.VectorSympy4D(x=x, y=y, z=z, t=t)

    # 2D + 3D.like(2D) = 2D
    assert v1 + v2.like(v1) == vector.VectorSympy2D(x=2 * x, y=2 * y)
    assert v2.like(v1) + v1 == vector.VectorSympy2D(x=2 * x, y=2 * y)
    # 2D + 4D.like(2D) = 2D
    assert v1 + v3.like(v1) == vector.VectorSympy2D(x=2 * x, y=2 * y)
    assert v3.like(v1) + v1 == vector.VectorSympy2D(x=2 * x, y=2 * y)
    # 3D + 2D.like(3D) = 3D
    assert v2 + v1.like(v2) == vector.VectorSympy3D(x=2 * x, y=2 * y, z=z)
    assert v1.like(v2) + v2 == vector.VectorSympy3D(x=2 * x, y=2 * y, z=z)
    # 3D + 4D.like(3D) = 3D
    assert v2 + v3.like(v2) == vector.VectorSympy3D(x=2 * x, y=2 * y, z=2 * z)
    assert v3.like(v2) + v2 == vector.VectorSympy3D(x=2 * x, y=2 * y, z=2 * z)
    # 4D + 2D.like(4D) = 4D
    assert v3 + v1.like(v3) == vector.VectorSympy4D(x=2 * x, y=2 * y, z=z, t=t)
    assert v1.like(v3) + v3 == vector.VectorSympy4D(x=2 * x, y=2 * y, z=z, t=t)
    # 4D + 3D.like(4D) = 4D
    assert v3 + v2.like(v3) == vector.VectorSympy4D(x=2 * x, y=2 * y, z=2 * z, t=t)
    assert v2.like(v3) + v3 == vector.VectorSympy4D(x=2 * x, y=2 * y, z=2 * z, t=t)

    v1 = vector.MomentumSympy2D(px=px, py=py)
    v2 = vector.MomentumSympy3D(px=px, py=py, pz=pz)
    v3 = vector.MomentumSympy4D(px=px, py=py, pz=pz, E=E)

    # order should not matter
    # 2D + 3D.like(2D) = 2D
    assert v1 + v2.like(v1) == vector.MomentumSympy2D(px=2 * px, py=2 * py)
    assert v2.like(v1) + v1 == vector.MomentumSympy2D(px=2 * px, py=2 * py)
    # 2D + 4D.like(2D) = 2D
    assert v1 + v3.like(v1) == vector.MomentumSympy2D(px=2 * px, py=2 * py)
    assert v3.like(v1) + v1 == vector.MomentumSympy2D(px=2 * px, py=2 * py)
    # 3D + 2D.like(3D) = 3D
    assert v2 + v1.like(v2) == vector.MomentumSympy3D(px=2 * px, py=2 * py, pz=pz)
    assert v1.like(v2) + v2 == vector.MomentumSympy3D(px=2 * px, py=2 * py, pz=pz)
    # 3D + 4D.like(3D) = 3D
    assert v2 + v3.like(v2) == vector.MomentumSympy3D(px=2 * px, py=2 * py, pz=2 * pz)
    assert v3.like(v2) + v2 == vector.MomentumSympy3D(px=2 * px, py=2 * py, pz=2 * pz)
    # 4D + 2D.like(4D) = 4D
    assert v3 + v1.like(v3) == vector.MomentumSympy4D(px=2 * px, py=2 * py, pz=pz, E=E)
    assert v1.like(v3) + v3 == vector.MomentumSympy4D(px=2 * px, py=2 * py, pz=pz, E=E)
    # 4D + 3D.like(4D) = 4D
    assert v3 + v2.like(v3) == vector.MomentumSympy4D(
        px=2 * px, py=2 * py, pz=2 * pz, E=E
    )
    assert v2.like(v3) + v3 == vector.MomentumSympy4D(
        px=2 * px, py=2 * py, pz=2 * pz, E=E
    )


def test_momentum_preservation():
    v1 = vector.MomentumSympy2D(px=px, py=py)
    v2 = vector.VectorSympy3D(x=x, y=y, z=z)
    v3 = vector.MomentumSympy4D(px=px, py=py, pz=pz, t=t)

    # momentum + generic = momentum
    # 2D + 3D.like(2D) = 2D
    assert isinstance(v1 + v2.like(v1), vector.MomentumSympy2D)
    assert isinstance(v2.like(v1) + v1, vector.MomentumSympy2D)
    # 2D + 4D.like(2D) = 2D
    assert isinstance(v1 + v3.like(v1), vector.MomentumSympy2D)
    assert isinstance(v3.like(v1) + v1, vector.MomentumSympy2D)
    # 3D + 2D.like(3D) = 3D
    assert isinstance(v2 + v1.like(v2), vector.MomentumSympy3D)
    assert isinstance(v1.like(v2) + v2, vector.MomentumSympy3D)
    # 3D + 4D.like(3D) = 3D
    assert isinstance(v2 + v3.like(v2), vector.MomentumSympy3D)
    assert isinstance(v3.like(v2) + v2, vector.MomentumSympy3D)
    # 4D + 2D.like(4D) = 4D
    assert isinstance(v3 + v1.like(v3), vector.MomentumSympy4D)
    assert isinstance(v1.like(v3) + v3, vector.MomentumSympy4D)
    # 4D + 3D.like(4D) = 4D
    assert isinstance(v3 + v2.like(v3), vector.MomentumSympy4D)
    assert isinstance(v2.like(v3) + v3, vector.MomentumSympy4D)
