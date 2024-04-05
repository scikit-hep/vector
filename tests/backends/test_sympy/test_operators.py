# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

sympy = pytest.importorskip("sympy")

pytestmark = pytest.mark.sympy

x, y, z = sympy.symbols("x y z")
nx, ny, nz = sympy.symbols("nx ny nz")

v1 = vector.VectorSympy2D(x=x, y=y)
v2 = vector.VectorSympy2D(x=nx, y=ny)


def test_eq():
    assert v1 == v1  # noqa: PLR0124
    assert not v1 == v2  # noqa: SIM201
    with pytest.raises(TypeError):
        v1.equal(v2.to_Vector3D())


def test_ne():
    assert not v1 != v1  # noqa: PLR0124,SIM202
    assert v1 != v2
    with pytest.raises(TypeError):
        v1.not_equal(v2.to_Vector3D())


def test_abs():
    assert abs(v1) == sympy.sqrt(x**2 + y**2)


def test_add():
    assert v1 + v2 == vector.VectorSympy2D(x=x + nx, y=y + ny)
    assert v1 + v2.to_Vector3D().like(v1) == vector.VectorSympy2D(x=x + nx, y=y + ny)
    with pytest.raises(TypeError):
        v1 + 5
    with pytest.raises(TypeError):
        5 + v1
    with pytest.raises(TypeError):
        v1 + v2.to_Vector3D()


def test_sub():
    assert v1 - v2 == vector.VectorSympy2D(x=x - nx, y=y - ny)
    with pytest.raises(TypeError):
        v1 - 5
    with pytest.raises(TypeError):
        5 - v1
    with pytest.raises(TypeError):
        v1 - v2.to_Vector3D()


def test_mul():
    assert v1 * 10 == vector.VectorSympy2D(x=x * 10, y=y * 10)
    assert 10 * v1 == vector.VectorSympy2D(x=10 * x, y=10 * y)
    with pytest.raises(TypeError):
        v1 * v2


def test_neg():
    assert -v1 == vector.VectorSympy2D(x=-x, y=-y)


def test_pos():
    assert +v1 == vector.VectorSympy2D(x=x, y=y)


def test_truediv():
    assert v1 / 10 == vector.VectorSympy2D(x=0.1 * x, y=0.1 * y)
    with pytest.raises(TypeError):
        10 / v1
    with pytest.raises(TypeError):
        v1 / v2


def test_pow():
    assert v1**2 == x**2 + y**2
    with pytest.raises(TypeError):
        2**v1
    with pytest.raises(TypeError):
        v1**v2


def test_matmul():
    assert v1 @ v2 == x * nx + y * ny
    assert v2 @ v1 == x * nx + y * ny
    with pytest.raises(TypeError):
        v1 @ 5
