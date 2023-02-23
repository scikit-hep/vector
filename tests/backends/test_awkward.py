# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

ak = pytest.importorskip("awkward")

pytestmark = pytest.mark.awkward


def test_dimension_conversion():
    # 2D -> 3D
    vec = vector.Array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.1}],
            [],
        ]
    )
    assert ak.all(vec.to_Vector3D(z=1).z == 1)
    assert ak.all(vec.to_Vector3D(eta=1).eta == 1)
    assert ak.all(vec.to_Vector3D(theta=1).theta == 1)

    assert ak.all(vec.to_Vector3D(z=1).x == vec.x)
    assert ak.all(vec.to_Vector3D(z=1).y == vec.y)

    # 2D -> 4D
    assert ak.all(vec.to_Vector4D(z=1, t=1).t == 1)
    assert ak.all(vec.to_Vector4D(z=1, t=1).z == 1)
    assert ak.all(vec.to_Vector4D(eta=1, t=1).eta == 1)
    assert ak.all(vec.to_Vector4D(eta=1, t=1).t == 1)
    assert ak.all(vec.to_Vector4D(theta=1, t=1).theta == 1)
    assert ak.all(vec.to_Vector4D(theta=1, t=1).t == 1)
    assert ak.all(vec.to_Vector4D(z=1, tau=1).z == 1)
    assert ak.all(vec.to_Vector4D(z=1, tau=1).tau == 1)
    assert ak.all(vec.to_Vector4D(eta=1, tau=1).eta == 1)
    assert ak.all(vec.to_Vector4D(eta=1, tau=1).tau == 1)
    assert ak.all(vec.to_Vector4D(theta=1, tau=1).theta == 1)
    assert ak.all(vec.to_Vector4D(theta=1, tau=1).tau == 1)

    assert ak.all(vec.to_Vector4D(z=1, t=1).x == vec.x)
    assert ak.all(vec.to_Vector4D(z=1, t=1).y == vec.y)

    # 3D -> 4D
    vec = vector.Array(
        [
            [{"x": 1, "y": 1.1, "z": 1.2}, {"x": 2, "y": 2.1, "z": 2.2}],
            [],
        ]
    )
    assert ak.all(vec.to_Vector4D(t=1).t == 1)
    assert ak.all(vec.to_Vector4D(tau=1).tau == 1)

    assert ak.all(vec.to_Vector4D(t=1).x == vec.x)
    assert ak.all(vec.to_Vector4D(t=1).y == vec.y)
    assert ak.all(vec.to_Vector4D(t=1).z == vec.z)


def test_type_checks():
    with pytest.raises(TypeError):
        vector.Array(
            [
                [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": [0.2]}],
                [],
            ]
        )

    with pytest.raises(TypeError):
        vector.Array(
            [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": complex(1, 2)}]
        )

    with pytest.raises(TypeError):
        vector.zip(
            [
                [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": 0.2}],
                [],
            ]
        )


def test_basic():
    array = vector.Array([[{"x": 1, "y": 2}], [], [{"x": 3, "y": 4}]])
    assert isinstance(array, vector.backends.awkward.VectorArray2D)
    assert array.x.tolist() == [[1], [], [3]]
    assert array.y.tolist() == [[2], [], [4]]
    assert array.rho.tolist() == [[2.23606797749979], [], [5]]
    (a,), (), (c,) = array.phi.tolist()
    assert a == pytest.approx(1.1071487177940904)
    assert c == pytest.approx(0.9272952180016122)
    assert isinstance(array[2, 0], vector.backends.awkward.VectorRecord2D)
    assert array[2, 0].rho == 5
    assert array.deltaphi(array).tolist() == [[0], [], [0]]

    array = vector.Array([[{"pt": 1, "phi": 2}], [], [{"pt": 3, "phi": 4}]])
    assert isinstance(array, vector.backends.awkward.MomentumArray2D)
    assert array.pt.tolist() == [[1], [], [3]]

    array = vector.Array(
        [
            [{"x": 1, "y": 2, "z": 3, "wow": 99}],
            [],
            [{"x": 4, "y": 5, "z": 6, "wow": 123}],
        ]
    )
    assert isinstance(array, vector.backends.awkward.VectorArray3D)
    assert array.wow.tolist() == [[99], [], [123]]


def test_rotateZ():
    array = vector.Array([[{"pt": 1, "phi": 0}], [], [{"pt": 2, "phi": 1}]])
    out = array.rotateZ(1)
    assert isinstance(out, vector.backends.awkward.MomentumArray2D)
    assert out.tolist() == [[{"rho": 1, "phi": 1}], [], [{"rho": 2, "phi": 2}]]

    array = vector.Array(
        [[{"x": 1, "y": 0, "wow": 99}], [], [{"x": 2, "y": 1, "wow": 123}]]
    )
    out = array.rotateZ(0.1)
    assert isinstance(out, vector.backends.awkward.VectorArray2D)
    assert out.wow.tolist() == [[99], [], [123]]


def test_projection():
    array = vector.Array(
        [
            [{"x": 1, "y": 2, "z": 3, "wow": 99}],
            [],
            [{"x": 4, "y": 5, "z": 6, "wow": 123}],
        ]
    )
    out = array.to_Vector2D()
    assert isinstance(out, vector.backends.awkward.VectorArray2D)
    assert out.tolist() == [
        [{"x": 1, "y": 2, "wow": 99}],
        [],
        [{"x": 4, "y": 5, "wow": 123}],
    ]

    out = array.to_Vector4D()
    assert isinstance(out, vector.backends.awkward.VectorArray4D)
    assert out.tolist() == [
        [{"x": 1, "y": 2, "z": 3, "t": 0, "wow": 99}],
        [],
        [{"x": 4, "y": 5, "z": 6, "t": 0, "wow": 123}],
    ]

    out = array.to_rhophietatau()
    assert isinstance(out, vector.backends.awkward.VectorArray4D)
    (a,), (), (c,) = out.tolist()
    assert a == pytest.approx(
        {
            "rho": 2.23606797749979,
            "phi": 1.1071487177940904,
            "eta": 1.1035868415601453,
            "tau": 0,
            "wow": 99,
        }
    )
    assert c == pytest.approx(
        {
            "rho": 6.4031242374328485,
            "phi": 0.8960553845713439,
            "eta": 0.8361481196083127,
            "tau": 0,
            "wow": 123,
        }
    )


def test_add():
    one = vector.Array(
        [[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}], [], [{"x": 3, "y": 3.3}]]
    )

    two = vector.Array(
        [{"x": 10, "y": 20}, {"x": 100, "y": 200}, {"x": 1000, "y": 2000}]
    )
    assert isinstance(one.add(two), vector.backends.awkward.VectorArray2D)
    assert isinstance(two.add(one), vector.backends.awkward.VectorArray2D)
    assert one.add(two).tolist() == [
        [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
        [],
        [{"x": 1003, "y": 2003.3}],
    ]
    assert two.add(one).tolist() == [
        [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
        [],
        [{"x": 1003, "y": 2003.3}],
    ]

    two = vector.array({"x": [10, 100, 1000], "y": [20, 200, 2000]})
    assert isinstance(one.add(two), vector.backends.awkward.VectorArray2D)
    assert isinstance(two.add(one), vector.backends.awkward.VectorArray2D)
    assert one.add(two).tolist() == [
        [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
        [],
        [{"x": 1003, "y": 2003.3}],
    ]
    assert two.add(one).tolist() == [
        [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
        [],
        [{"x": 1003, "y": 2003.3}],
    ]

    two = vector.obj(x=10, y=20)
    assert isinstance(one.add(two), vector.backends.awkward.VectorArray2D)
    assert isinstance(two.add(one), vector.backends.awkward.VectorArray2D)
    assert one.add(two).tolist() == [
        [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
        [],
        [{"x": 13, "y": 23.3}],
    ]
    assert two.add(one).tolist() == [
        [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
        [],
        [{"x": 13, "y": 23.3}],
    ]


def test_ufuncs():
    array = vector.Array(
        [[{"x": 3, "y": 4}, {"x": 5, "y": 12}], [], [{"x": 8, "y": 15}]]
    )
    assert abs(array).tolist() == [[5, 13], [], [17]]
    assert (array**2).tolist() == [[5**2, 13**2], [], [17**2]]

    one = vector.Array(
        [[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}], [], [{"x": 3, "y": 3.3}]]
    )
    two = vector.Array(
        [{"x": 10, "y": 20}, {"x": 100, "y": 200}, {"x": 1000, "y": 2000}]
    )
    assert (one + two).tolist() == [
        [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
        [],
        [{"x": 1003, "y": 2003.3}],
    ]

    assert (one * 10).tolist() == [
        [{"x": 10, "y": 11}, {"x": 20, "y": 22}],
        [],
        [{"x": 30, "y": 33}],
    ]


def test_zip():
    v = vector.zip({"x": [[], [1]], "y": [[], [1]]})
    assert isinstance(v, vector.backends.awkward.VectorArray2D)
    assert isinstance(v[1], vector.backends.awkward.VectorArray2D)
    assert isinstance(v[1, 0], vector.backends.awkward.VectorRecord2D)
    assert v.tolist() == [[], [{"x": 1, "y": 1}]]
    assert v.x.tolist() == [[], [1]]
    assert v.y.tolist() == [[], [1]]
