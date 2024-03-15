# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import importlib.metadata
import numbers

import numpy as np
import packaging.version
import pytest

import vector
from vector import VectorObject2D
from vector.backends.awkward import (
    MomentumAwkward2D,
    MomentumAwkward3D,
    MomentumAwkward4D,
)

ak = pytest.importorskip("awkward")

pytestmark = pytest.mark.awkward


# Record reducers were added before awkward==2.2.3, but had some bugs.
awkward_without_record_reducers = packaging.version.Version(
    importlib.metadata.version("awkward")
) < packaging.version.Version("2.2.3")


def test_dimension_conversion():
    # 2D -> 3D
    vec = vector.Array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.1}],
            [],
        ]
    )
    # test alias
    assert ak.all(vec.to_3D(z=1).z == 1)
    assert ak.all(vec.to_3D(eta=1).eta == 1)
    assert ak.all(vec.to_3D(theta=1).theta == 1)

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

    # test alias
    assert ak.all(vec.to_4D(z=1, t=1).x == vec.x)
    assert ak.all(vec.to_4D(z=1, t=1).y == vec.y)

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

    # check if momentum coords work
    vec = vector.Array(
        [
            [{"px": 1, "py": 1.1}, {"px": 2, "py": 2.1}],
            [],
        ]
    )
    assert ak.all(vec.to_Vector3D(pz=1).pz == 1)

    assert ak.all(vec.to_Vector4D(pz=1, m=1).pz == 1)
    assert ak.all(vec.to_Vector4D(pz=1, m=1).m == 1)
    assert ak.all(vec.to_Vector4D(pz=1, mass=1).mass == 1)
    assert ak.all(vec.to_Vector4D(pz=1, M=1).M == 1)
    assert ak.all(vec.to_Vector4D(pz=1, e=1).e == 1)
    assert ak.all(vec.to_Vector4D(pz=1, energy=1).energy == 1)
    assert ak.all(vec.to_Vector4D(pz=1, E=1).E == 1)

    vec = vector.Array(
        [
            [{"px": 1, "py": 1.1, "pz": 1.2}, {"px": 2, "py": 2.1, "pz": 2.2}],
            [],
        ]
    )
    assert ak.all(vec.to_Vector4D(m=1).m == 1)
    assert ak.all(vec.to_Vector4D(mass=1).mass == 1)
    assert ak.all(vec.to_Vector4D(M=1).M == 1)
    assert ak.all(vec.to_Vector4D(e=1).e == 1)
    assert ak.all(vec.to_Vector4D(energy=1).energy == 1)
    assert ak.all(vec.to_Vector4D(E=1).E == 1)


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


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers not implemented before awkward v2",
)
def test_sum_2d():
    v = vector.Array(
        [
            [
                {"rho": 1.0, "phi": 0.1},
                {"rho": 4.0, "phi": 0.2},
            ],
            [
                {"rho": 1.0, "phi": 0.3},
                {"rho": 4.0, "phi": 0.4},
                {"rho": 1.0, "phi": 0.1},
            ],
        ]
    )
    assert ak.almost_equal(
        ak.sum(v, axis=0, keepdims=True),
        vector.Array(
            [
                [
                    {"x": 1.950340654403632, "y": 0.3953536233081677},
                    {"x": 7.604510287376507, "y": 2.3523506924148467},
                    {"x": 0.9950041652780258, "y": 0.09983341664682815},
                ]
            ]
        ),
    )
    assert ak.almost_equal(
        ak.sum(v, axis=0, keepdims=False),
        vector.Array(
            [
                {"x": 1.950340654403632, "y": 0.3953536233081677},
                {"x": 7.604510287376507, "y": 2.3523506924148467},
                {"x": 0.9950041652780258, "y": 0.09983341664682815},
            ]
        ),
    )
    assert ak.almost_equal(
        ak.sum(v, axis=1, keepdims=True),
        ak.to_regular(
            vector.Array(
                [
                    [{"x": 4.915270, "y": 0.89451074}],
                    [{"x": 5.63458463, "y": 1.95302699}],
                ]
            )
        ),
    )
    assert ak.almost_equal(
        ak.sum(v, axis=1, keepdims=False),
        vector.Array(
            [
                {"x": 4.915270, "y": 0.89451074},
                {"x": 5.63458463, "y": 1.95302699},
            ]
        ),
    )
    assert ak.almost_equal(
        ak.sum(v.mask[[False, True]], axis=1),
        vector.Array(
            [
                {"x": 5.63458463, "y": 1.95302699},
            ]
        )[[None, 0]],
    )


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers not implemented before awkward v2",
)
def test_sum_3d():
    v = vector.Array(
        [
            [
                {"x": 1, "y": 2, "z": 3},
                {"x": 4, "y": 5, "z": 6},
            ],
            [
                {"x": 1, "y": 2, "z": 3},
                {"x": 4, "y": 5, "z": 6},
                {"x": 1, "y": 1, "z": 1},
            ],
        ]
    )
    assert ak.sum(v, axis=0, keepdims=True).to_list() == [
        [{"x": 2, "y": 4, "z": 6}, {"x": 8, "y": 10, "z": 12}, {"x": 1, "y": 1, "z": 1}]
    ]
    assert ak.sum(v, axis=0, keepdims=False).to_list() == [
        {"x": 2, "y": 4, "z": 6},
        {"x": 8, "y": 10, "z": 12},
        {"x": 1, "y": 1, "z": 1},
    ]
    assert ak.sum(v, axis=1, keepdims=True).to_list() == [
        [{"x": 5, "y": 7, "z": 9}],
        [{"x": 6, "y": 8, "z": 10}],
    ]
    assert ak.sum(v, axis=1, keepdims=False).to_list() == [
        {"x": 5, "y": 7, "z": 9},
        {"x": 6, "y": 8, "z": 10},
    ]
    assert ak.sum(v.mask[[False, True]], axis=1).tolist() == [
        None,
        {"x": 6, "y": 8, "z": 10},
    ]


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers not implemented before awkward v2",
)
def test_sum_4d():
    v = vector.Array(
        [
            [
                {"x": 1, "y": 2, "z": 3, "t": 4},
                {"x": 4, "y": 5, "z": 6, "t": 2},
                {"x": 0, "y": 0, "z": 0, "t": 3},
            ],
            [
                {"x": 1, "y": 2, "z": 3, "t": 8},
                {"x": 4, "y": 5, "z": 6, "t": 0},
                {"x": 1, "y": 1, "z": 1, "t": 0},
            ],
        ]
    )
    assert ak.sum(v, axis=0, keepdims=True).to_list() == [
        [
            {"t": 12, "z": 6, "x": 2, "y": 4},
            {"t": 2, "z": 12, "x": 8, "y": 10},
            {"t": 3, "z": 1, "x": 1, "y": 1},
        ]
    ]
    assert ak.sum(v, axis=0, keepdims=False).to_list() == [
        {"t": 12, "z": 6, "x": 2, "y": 4},
        {"t": 2, "z": 12, "x": 8, "y": 10},
        {"t": 3, "z": 1, "x": 1, "y": 1},
    ]
    assert ak.sum(v, axis=1, keepdims=True).to_list() == [
        [{"t": 9, "z": 9, "x": 5, "y": 7}],
        [{"t": 8, "z": 10, "x": 6, "y": 8}],
    ]
    assert ak.sum(v, axis=1, keepdims=False).to_list() == [
        {"t": 9, "z": 9, "x": 5, "y": 7},
        {"t": 8, "z": 10, "x": 6, "y": 8},
    ]
    assert ak.sum(v.mask[[False, True]], axis=1).tolist() == [
        None,
        {"t": 8, "z": 10, "x": 6, "y": 8},
    ]


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers not implemented before awkward v2",
)
def test_count_nonzero_2d():
    v = vector.Array(
        [
            [
                {"rho": 1.0, "phi": 0.1},
                {"rho": 4.0, "phi": 0.2},
                {"rho": 0.0, "phi": 0.0},
            ],
            [
                {"rho": 1.0, "phi": 0.3},
                {"rho": 4.0, "phi": 0.4},
                {"rho": 1.0, "phi": 0.1},
            ],
        ]
    )
    assert ak.count_nonzero(v, axis=1).tolist() == [2, 3]
    assert ak.count_nonzero(v, axis=1, keepdims=True).tolist() == [[2], [3]]
    assert ak.count_nonzero(v, axis=0).tolist() == [2, 2, 1]
    assert ak.count_nonzero(v, axis=0, keepdims=True).tolist() == [[2, 2, 1]]


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers not implemented before awkward v2",
)
def test_count_nonzero_3d():
    v = vector.Array(
        [
            [
                {"x": 1.0, "y": 2.0, "theta": 0.1},
                {"x": 4.0, "y": 5.0, "theta": 0.2},
                {"x": 0.0, "y": 0.0, "theta": 0.0},
            ],
            [
                {"x": 1.0, "y": 2.0, "theta": 0.6},
                {"x": 4.0, "y": 5.0, "theta": 1.3},
                {"x": 1.0, "y": 1.0, "theta": 1.9},
            ],
        ]
    )
    assert ak.count_nonzero(v, axis=1).tolist() == [2, 3]
    assert ak.count_nonzero(v, axis=1, keepdims=True).tolist() == [[2], [3]]
    assert ak.count_nonzero(v, axis=0).tolist() == [2, 2, 1]
    assert ak.count_nonzero(v, axis=0, keepdims=True).tolist() == [[2, 2, 1]]


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers not implemented before awkward v2",
)
def test_count_nonzero_4d():
    v = vector.Array(
        [
            [
                {"x": 1.0, "y": 2.0, "z": 3.0, "t": 4.0},
                {"x": 4.0, "y": 5.0, "z": 6.0, "t": 2.0},
                {"x": 0.0, "y": 0.0, "z": 0.0, "t": 3.0},
            ],
            [
                {"x": 1.0, "y": 2.0, "z": 3.0, "t": 8.0},
                {"x": 4.0, "y": 5.0, "z": 6.0, "t": 0.0},
                {"x": 1.0, "y": 1.0, "z": 1.0, "t": 0.0},
            ],
        ]
    )
    assert ak.count_nonzero(v, axis=1).tolist() == [3, 3]
    assert ak.count_nonzero(v, axis=1, keepdims=True).tolist() == [[3], [3]]
    assert ak.count_nonzero(v, axis=0).tolist() == [2, 2, 2]
    assert ak.count_nonzero(v, axis=0, keepdims=True).tolist() == [[2, 2, 2]]

    v2 = vector.Array(
        [
            [
                {"x": 1, "y": 2, "z": 3, "t": 1},
                {"x": 4, "y": 5, "z": 6, "t": 2},
                {"x": 0, "y": 0, "z": 0, "t": 2},
            ],
            [
                {"x": 1, "y": 2, "z": 3, "t": 0},
                {"x": 4, "y": 5, "z": 6, "t": 1},
                {"x": 0, "y": 0, "z": 0, "t": 0},
            ],
        ]
    )
    assert ak.count_nonzero(v2, axis=1).tolist() == [3, 2]
    assert ak.count_nonzero(v2, axis=1, keepdims=True).tolist() == [[3], [2]]
    assert ak.count_nonzero(v2, axis=0).tolist() == [2, 2, 1]
    assert ak.count_nonzero(v2, axis=0, keepdims=True).tolist() == [[2, 2, 1]]


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers not implemented before awkward v2",
)
def test_count_2d():
    v = vector.Array(
        [
            [
                {"x": 1, "y": 2},
                {"x": 4, "y": 5},
                {"x": 0, "y": 0},
            ],
            [
                {"x": 1, "y": 2},
                {"x": 4, "y": 5},
                {"x": 1, "y": 1},
            ],
        ]
    )
    assert ak.count(v, axis=1).to_list() == [3, 3]
    assert ak.count(v, axis=1, keepdims=True).to_list() == [[3], [3]]
    assert ak.count(v, axis=0).to_list() == [2, 2, 2]
    assert ak.count(v, axis=0, keepdims=True).to_list() == [[2, 2, 2]]
    assert ak.count(v.mask[[False, True]], axis=1).tolist() == [
        None,
        3,
    ]


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers not implemented before awkward v2",
)
def test_count_3d():
    v = vector.Array(
        [
            [
                {"x": 1, "y": 2, "z": 3},
                {"x": 4, "y": 5, "z": 6},
                {"x": 0, "y": 0, "z": 0},
            ],
            [
                {"x": 1, "y": 2, "z": 3},
                {"x": 4, "y": 5, "z": 6},
                {"x": 1, "y": 1, "z": 1},
            ],
        ]
    )
    assert ak.count(v, axis=1).to_list() == [3, 3]
    assert ak.count(v, axis=1, keepdims=True).to_list() == [[3], [3]]
    assert ak.count(v, axis=0).to_list() == [2, 2, 2]
    assert ak.count(v, axis=0, keepdims=True).to_list() == [[2, 2, 2]]
    assert ak.count(v.mask[[False, True]], axis=1).tolist() == [
        None,
        3,
    ]


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers not implemented before awkward v2",
)
def test_count_4d():
    v = vector.Array(
        [
            [
                {"x": 1, "y": 2, "z": 3, "t": 9},
                {"x": 4, "y": 5, "z": 6, "t": 9},
                {"x": 0, "y": 0, "z": 0, "t": 9},
            ],
            [
                {"x": 1, "y": 2, "z": 3, "t": 9},
                {"x": 4, "y": 5, "z": 6, "t": 9},
                {"x": 1, "y": 1, "z": 1, "t": 9},
            ],
        ]
    )
    assert ak.count(v, axis=1).to_list() == [3, 3]
    assert ak.count(v, axis=1, keepdims=True).to_list() == [[3], [3]]
    assert ak.count(v, axis=0).to_list() == [2, 2, 2]
    assert ak.count(v, axis=0, keepdims=True).to_list() == [[2, 2, 2]]
    assert ak.count(v.mask[[False, True]], axis=1).tolist() == [
        None,
        3,
    ]


def test_like():
    v1 = vector.zip(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
        },
    )
    v2 = vector.zip(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
            "z": [5.0, 1.0, 1.0],
        },
    )
    v3 = vector.zip(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
            "z": [5.0, 1.0, 1.0],
            "t": [16.0, 31.0, 46.0],
        },
    )

    v1_v2 = vector.zip(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
        },
    )
    v2_v1 = vector.zip(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
            "z": [5.0, 1.0, 1.0],
        },
    )
    v2_v3 = vector.zip(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
            "z": [10.0, 2.0, 2.0],
        },
    )
    v3_v2 = vector.zip(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
            "z": [10.0, 2.0, 2.0],
            "t": [16.0, 31.0, 46.0],
        },
    )
    v1_v3 = vector.zip(
        {
            "x": [20.0, 40.0, 60.0],
            "y": [-20.0, 40.0, 60.0],
            "z": [5.0, 1.0, 1.0],
            "t": [16.0, 31.0, 46.0],
        },
    )

    with pytest.raises(TypeError):
        v1 + v2
    with pytest.raises(TypeError):
        v2 + v3
    with pytest.raises(TypeError):
        v1 + v3
    with pytest.raises(TypeError):
        v1 - v2
    with pytest.raises(TypeError):
        v2 - v3
    with pytest.raises(TypeError):
        v1 - v3
    with pytest.raises(TypeError):
        v1.equal(v2)
    with pytest.raises(TypeError):
        v2.equal(v3)
    with pytest.raises(TypeError):
        v1.equal(v3)
    with pytest.raises(TypeError):
        v1.not_equal(v2)
    with pytest.raises(TypeError):
        v2.not_equal(v3)
    with pytest.raises(TypeError):
        v1.not_equal(v3)
    with pytest.raises(TypeError):
        v1.dot(v2)
    with pytest.raises(TypeError):
        v2.dot(v3)
    with pytest.raises(TypeError):
        v1.dot(v3)

    # 2D + 3D.like(2D) = 2D
    assert ak.all(v1 + v2.like(v1) == v1_v2)
    assert ak.all(v2.like(v1) + v1 == v1_v2)
    # 2D + 4D.like(2D) = 2D
    assert ak.all(v1 + v3.like(v1) == v1_v2)
    assert ak.all(v3.like(v1) + v1 == v1_v2)
    # 3D + 2D.like(3D) = 3D
    assert ak.all(v2 + v1.like(v2) == v2_v1)
    assert ak.all(v1.like(v2) + v2 == v2_v1)
    # 3D + 4D.like(3D) = 3D
    assert ak.all(v2 + v3.like(v2) == v2_v3)
    assert ak.all(v3.like(v2) + v2 == v2_v3)
    # 4D + 2D.like(4D) = 4D
    assert ak.all(v3 + v1.like(v3) == v1_v3)
    assert ak.all(v1.like(v3) + v3 == v1_v3)
    # 4D + 3D.like(4D) = 4D
    assert ak.all(v3 + v2.like(v3) == v3_v2)
    assert ak.all(v2.like(v3) + v3 == v3_v2)

    v1 = vector.zip(
        {
            "px": [10.0, 20.0, 30.0],
            "py": [-10.0, 20.0, 30.0],
        },
    )
    v2 = vector.zip(
        {
            "px": [10.0, 20.0, 30.0],
            "py": [-10.0, 20.0, 30.0],
            "pz": [5.0, 1.0, 1.0],
        },
    )
    v3 = vector.zip(
        {
            "px": [10.0, 20.0, 30.0],
            "py": [-10.0, 20.0, 30.0],
            "pz": [5.0, 1.0, 1.0],
            "t": [16.0, 31.0, 46.0],
        },
    )

    # 2D + 3D.like(2D) = 2D
    assert ak.all(v1 + v2.like(v1) == v1_v2)
    assert ak.all(v2.like(v1) + v1 == v1_v2)
    # 2D + 4D.like(2D) = 2D
    assert ak.all(v1 + v3.like(v1) == v1_v2)
    assert ak.all(v3.like(v1) + v1 == v1_v2)
    # 3D + 2D.like(3D) = 3D
    assert ak.all(v2 + v1.like(v2) == v2_v1)
    assert ak.all(v1.like(v2) + v2 == v2_v1)
    # 3D + 4D.like(3D) = 3D
    assert ak.all(v2 + v3.like(v2) == v2_v3)
    assert ak.all(v3.like(v2) + v2 == v2_v3)
    # 4D + 2D.like(4D) = 4D
    assert ak.all(v3 + v1.like(v3) == v1_v3)
    assert ak.all(v1.like(v3) + v3 == v1_v3)
    # 4D + 3D.like(4D) = 4D
    assert ak.all(v3 + v2.like(v3) == v3_v2)
    assert ak.all(v2.like(v3) + v3 == v3_v2)


def test_handler_of():
    numpy_vec = vector.array(
        {
            "x": [1.1, 1.2, 1.3, 1.4, 1.5],
            "y": [2.1, 2.2, 2.3, 2.4, 2.5],
            "z": [3.1, 3.2, 3.3, 3.4, 3.5],
        }
    )
    awkward_vec2 = vector.zip(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [-1.0, 2.0, 3.0],
            "z": [5.0, 10.0, 15.0],
            "t": [16.0, 31.0, 46.0],
        },
    )
    object_vec = VectorObject2D.from_xy(1.0, 1.0)
    protocol = vector._methods._handler_of(object_vec, numpy_vec, awkward_vec2)
    # chooses awkward backend
    assert all(protocol == awkward_vec2)


def test_momentum_coordinate_transforms():
    awkward_vec = vector.zip(
        {
            "px": [1.0, 2.0, 3.0],
            "py": [-1.0, 2.0, 3.0],
        },
    )

    for t1 in "pxpy", "ptphi":
        for t2 in "pz", "eta", "theta":
            for t3 in "mass", "energy":
                transformed_object = getattr(awkward_vec, "to_" + t1)()
                assert isinstance(
                    transformed_object, vector.backends.awkward.MomentumAwkward2D
                )
                assert hasattr(transformed_object, t1[:2])
                assert hasattr(transformed_object, t1[2:])

                transformed_object = getattr(awkward_vec, "to_" + t1 + t2)()
                assert isinstance(
                    transformed_object, vector.backends.awkward.MomentumAwkward3D
                )
                assert hasattr(transformed_object, t1[:2])
                assert hasattr(transformed_object, t1[2:])
                assert hasattr(transformed_object, t2)

                transformed_object = getattr(awkward_vec, "to_" + t1 + t2 + t3)()
                assert isinstance(
                    transformed_object, vector.backends.awkward.MomentumAwkward4D
                )
                assert hasattr(transformed_object, t1[:2])
                assert hasattr(transformed_object, t1[2:])
                assert hasattr(transformed_object, t2)
                assert hasattr(transformed_object, t3)


def test_momentum_preservation():
    v1 = vector.zip(
        {
            "px": [10.0, 20.0, 30.0],
            "py": [-10.0, 20.0, 30.0],
        },
    )
    v2 = vector.zip(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [-10.0, 20.0, 30.0],
            "z": [5.0, 1.0, 1.0],
        },
    )

    v3 = vector.zip(
        {
            "px": [10.0, 20.0, 30.0],
            "py": [-10.0, 20.0, 30.0],
            "pz": [5.0, 1.0, 1.0],
            "t": [16.0, 31.0, 46.0],
        },
    )

    # momentum + generic = momentum
    # 2D + 3D.like(2D) = 2D
    assert isinstance(v1 + v2.like(v1), MomentumAwkward2D)
    assert isinstance(v2.like(v1) + v1, MomentumAwkward2D)
    # 2D + 4D.like(2D) = 2D
    assert isinstance(v1 + v3.like(v1), MomentumAwkward2D)
    assert isinstance(v3.like(v1) + v1, MomentumAwkward2D)
    # 3D + 2D.like(3D) = 3D
    assert isinstance(v2 + v1.like(v2), MomentumAwkward3D)
    assert isinstance(v1.like(v2) + v2, MomentumAwkward3D)
    # 3D + 4D.like(3D) = 3D
    assert isinstance(v2 + v3.like(v2), MomentumAwkward3D)
    assert isinstance(v3.like(v2) + v2, MomentumAwkward3D)
    # 4D + 2D.like(4D) = 4D
    assert isinstance(v3 + v1.like(v3), MomentumAwkward4D)
    assert isinstance(v1.like(v3) + v3, MomentumAwkward4D)
    # 4D + 3D.like(4D) = 4D
    assert isinstance(v3 + v2.like(v3), MomentumAwkward4D)
    assert isinstance(v2.like(v3) + v3, MomentumAwkward4D)


def test_subclass_fields():
    @ak.mixin_class(vector.backends.awkward.behavior)
    class TwoVector(MomentumAwkward2D):
        pass

    @ak.mixin_class(vector.backends.awkward.behavior)
    class ThreeVector(MomentumAwkward3D):
        pass

    @ak.mixin_class(vector.backends.awkward.behavior)
    class LorentzVector(MomentumAwkward4D):
        @ak.mixin_class_method(np.divide, {numbers.Number})
        def divide(self, factor):
            return self.scale(1 / factor)

    LorentzVectorArray.ProjectionClass2D = TwoVectorArray  # noqa: F821
    LorentzVectorArray.ProjectionClass3D = ThreeVectorArray  # noqa: F821
    LorentzVectorArray.ProjectionClass4D = LorentzVectorArray  # noqa: F821
    LorentzVectorArray.MomentumClass = LorentzVectorArray  # noqa: F821

    vec = ak.zip(
        {
            "pt": [[1, 2], [], [3], [4]],
            "eta": [[1.2, 1.4], [], [1.6], [3.4]],
            "phi": [[0.3, 0.4], [], [0.5], [0.6]],
            "energy": [[50, 51], [], [52], [60]],
        },
        with_name="LorentzVector",
        behavior=vector.backends.awkward.behavior,
    )

    assert vec.like(vector.obj(x=1, y=2)).fields == ["rho", "phi"]
    assert vec.like(vector.obj(x=1, y=2, z=3)).fields == ["rho", "phi", "eta"]
    assert (vec / 2).fields == ["rho", "phi", "eta", "t"]
