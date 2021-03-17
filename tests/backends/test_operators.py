# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy
import pytest

import vector

v1 = vector.obj(x=1, y=5)
a1 = vector.array({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})

v2 = vector.obj(x=10, y=20)
a2 = vector.array({"x": [10, 100, 1000, 10000], "y": [20, 200, 2000, 20000]})


def test_eq():
    assert v1 == v1
    assert not v1 == v2
    assert (a1 == a1).all()
    assert not (a1 == a2).any()
    assert (v1 == a1).any()
    assert not (v1 == a1).all()
    assert (a1 == v1).any()
    assert not (a1 == v1).all()


def test_ne():
    assert not v1 != v1
    assert v1 != v2
    assert not (a1 != a1).any()
    assert (a1 != a2).all()
    assert (v1 != a1).any()
    assert not (v1 != a1).all()
    assert (a1 != v1).any()
    assert not (a1 != v1).all()


def test_abs():
    assert abs(v1) == pytest.approx(numpy.sqrt(1 ** 2 + 5 ** 2))
    assert numpy.allclose(
        abs(a1),
        numpy.sqrt(numpy.array([1, 2, 3, 4]) ** 2 + numpy.array([5, 6, 7, 8]) ** 2),
    )


def test_add():
    assert v1 + v2 == vector.obj(x=11, y=25)
    assert numpy.allclose(
        a1 + a2,
        vector.array({"x": [11, 102, 1003, 10004], "y": [25, 206, 2007, 20008]}),
    )
    assert numpy.allclose(
        v1 + a2,
        vector.array({"x": [11, 101, 1001, 10001], "y": [25, 205, 2005, 20005]}),
    )
    assert numpy.allclose(
        a2 + v1,
        vector.array({"x": [11, 101, 1001, 10001], "y": [25, 205, 2005, 20005]}),
    )
    with pytest.raises(TypeError):
        v1 + 5
    with pytest.raises(TypeError):
        5 + v1


def test_sub():
    assert v1 - v2 == vector.obj(x=-9, y=-15)
    assert numpy.allclose(
        a1 - a2,
        vector.array({"x": [-9, -98, -997, -9996], "y": [-15, -194, -1993, -19992]}),
    )
    assert numpy.allclose(
        v1 - a2,
        vector.array({"x": [-9, -99, -999, -9999], "y": [-15, -195, -1995, -19995]}),
    )
    assert numpy.allclose(
        a2 - v1,
        vector.array({"x": [9, 99, 999, 9999], "y": [15, 195, 1995, 19995]}),
    )
    with pytest.raises(TypeError):
        v1 - 5
    with pytest.raises(TypeError):
        5 - v1


def test_mul():
    assert v1 * 10 == vector.obj(x=10, y=50)
    assert 10 * v1 == vector.obj(x=10, y=50)
    assert numpy.allclose(
        a1 * 10, vector.array({"x": [10, 20, 30, 40], "y": [50, 60, 70, 80]})
    )
    assert numpy.allclose(
        10 * a1, vector.array({"x": [10, 20, 30, 40], "y": [50, 60, 70, 80]})
    )
    with pytest.raises(TypeError):
        v1 * v2
    with pytest.raises(TypeError):
        a1 * a2


def test_neg():
    assert -v1 == vector.obj(x=-1, y=-5)
    assert numpy.allclose(
        -a1, vector.array({"x": [-1, -2, -3, -4], "y": [-5, -6, -7, -8]})
    )


def test_pos():
    assert +v1 == vector.obj(x=1, y=5)
    assert numpy.allclose(+a1, vector.array({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]}))


def test_truediv():
    assert v1 / 10 == vector.obj(x=0.1, y=0.5)
    with pytest.raises(TypeError):
        10 / v1
    assert numpy.allclose(
        a1 / 10, vector.array({"x": [0.1, 0.2, 0.3, 0.4], "y": [0.5, 0.6, 0.7, 0.8]})
    )
    with pytest.raises(TypeError):
        10 / a1
    with pytest.raises(TypeError):
        v1 / v2
    with pytest.raises(TypeError):
        a1 / a2


def test_pow():
    assert v1 ** 2 == pytest.approx(1 ** 2 + 5 ** 2)
    with pytest.raises(TypeError):
        2 ** v1
    assert numpy.allclose(
        a1 ** 2,
        numpy.array(
            [1 ** 2 + 5 ** 2, 2 ** 2 + 6 ** 2, 3 ** 2 + 7 ** 2, 4 ** 2 + 8 ** 2]
        ),
    )
    with pytest.raises(TypeError):
        2 ** a1
    with pytest.raises(TypeError):
        v1 ** v2
    with pytest.raises(TypeError):
        a1 ** a2


def test_matmul():
    assert v1 @ v2 == pytest.approx(1 * 10 + 5 * 20)
    assert v2 @ v1 == pytest.approx(1 * 10 + 5 * 20)
    assert numpy.allclose(
        a1 @ a2,
        numpy.array(
            [
                1 * 10 + 5 * 20,
                2 * 100 + 6 * 200,
                3 * 1000 + 7 * 2000,
                4 * 10000 + 8 * 20000,
            ]
        ),
    )
    assert numpy.allclose(
        a2 @ a1,
        numpy.array(
            [
                1 * 10 + 5 * 20,
                2 * 100 + 6 * 200,
                3 * 1000 + 7 * 2000,
                4 * 10000 + 8 * 20000,
            ]
        ),
    )
    assert numpy.allclose(
        v1 @ a2,
        numpy.array(
            [
                1 * 10 + 5 * 20,
                1 * 100 + 5 * 200,
                1 * 1000 + 5 * 2000,
                1 * 10000 + 5 * 20000,
            ]
        ),
    )
    assert numpy.allclose(
        a2 @ v1,
        numpy.array(
            [
                1 * 10 + 5 * 20,
                1 * 100 + 5 * 200,
                1 * 1000 + 5 * 2000,
                1 * 10000 + 5 * 20000,
            ]
        ),
    )
    with pytest.raises(TypeError):
        v1 @ 5
    with pytest.raises(TypeError):
        a1 @ 5
