# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import functools
import importlib.metadata
import numbers

import numpy as np
import packaging.version
import pytest

import vector
from vector import VectorObject2D

ak = pytest.importorskip("awkward")

pytestmark = pytest.mark.awkward


# Record reducers were added before awkward==2.2.3, but had some bugs.
awkward_without_record_reducers = packaging.version.Version(
    importlib.metadata.version("awkward")
) < packaging.version.Version("2.2.3")


def _assert(bool1, bool2, backend):
    # this is a helper function that differentiates between backends:
    # - 'typetracer' backend:
    #   this method does nothing (no-op). typetracers can't be compared as they do not have numerical values.
    #   this is useful, because the expression that goes into this function (bool1 & bool2) will still be computed,
    #   i.e. this makes sure that the computation does (at least) not raise an error.
    # - other backends:
    #   convert awkward arrays to lists and compare their values with python's builtin list comparison
    if backend != "typetracer":
        if isinstance(bool1, ak.Array):
            bool1 = bool1.tolist()
        if isinstance(bool2, ak.Array):
            bool2 = bool2.tolist()
        else:
            assert bool1 == bool2


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_dimension_conversion(backend):
    assert_backend = functools.partial(_assert, backend=backend)

    # 2D -> 3D
    vec = vector.Array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.1}],
            [],
        ],
        backend=backend,
    )

    # test alias
    assert_backend(ak.all(vec.to_3D(z=1).z == 1), True)
    assert_backend(ak.all(vec.to_3D(eta=1).eta == 1), True)
    assert_backend(ak.all(vec.to_3D(theta=1).theta == 1), True)

    assert_backend(vec.to_Vector3D(z=1).x, vec.x)
    assert_backend(vec.to_Vector3D(z=1).y, vec.y)

    # 2D -> 4D
    assert_backend(ak.all(vec.to_Vector4D(z=1, t=1).t == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(z=1, t=1).z == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(eta=1, t=1).eta == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(eta=1, t=1).t == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(theta=1, t=1).theta == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(theta=1, t=1).t == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(z=1, tau=1).z == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(z=1, tau=1).tau == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(eta=1, tau=1).eta == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(eta=1, tau=1).tau == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(theta=1, tau=1).theta == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(theta=1, tau=1).tau == 1), True)

    # test alias
    assert_backend(vec.to_4D(z=1, t=1).x, vec.x)
    assert_backend(vec.to_4D(z=1, t=1).y, vec.y)

    # 3D -> 4D
    vec = vector.Array(
        [
            [{"x": 1, "y": 1.1, "z": 1.2}, {"x": 2, "y": 2.1, "z": 2.2}],
            [],
        ],
        backend=backend,
    )
    assert_backend(ak.all(vec.to_Vector4D(t=1).t == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(tau=1).tau == 1), True)

    assert_backend(vec.to_Vector4D(t=1).x, vec.x)
    assert_backend(vec.to_Vector4D(t=1).y, vec.y)
    assert_backend(vec.to_Vector4D(t=1).z, vec.z)

    # check if momentum coords work
    vec = vector.Array(
        [
            [{"px": 1, "py": 1.1}, {"px": 2, "py": 2.1}],
            [],
        ],
        backend=backend,
    )
    assert_backend(ak.all(vec.to_Vector3D(pz=1).pz == 1), True)

    assert_backend(ak.all(vec.to_Vector4D(pz=1, m=1).pz == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(pz=1, m=1).m == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(pz=1, mass=1).mass == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(pz=1, M=1).M == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(pz=1, e=1).e == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(pz=1, energy=1).energy == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(pz=1, E=1).E == 1), True)

    vec = vector.Array(
        [
            [{"px": 1, "py": 1.1, "pz": 1.2}, {"px": 2, "py": 2.1, "pz": 2.2}],
            [],
        ],
        backend=backend,
    )
    assert_backend(ak.all(vec.to_Vector4D(m=1).m == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(mass=1).mass == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(M=1).M == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(e=1).e == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(energy=1).energy == 1), True)
    assert_backend(ak.all(vec.to_Vector4D(E=1).E == 1), True)


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_type_checks(backend):
    with pytest.raises(TypeError):
        vector.Array(
            [
                [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": [0.2]}],
                [],
            ],
            backend=backend,
        )

    with pytest.raises(TypeError):
        vector.Array(
            [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": complex(1, 2)}],
            backend=backend,
        )

    with pytest.raises(TypeError):
        vector.zip(
            [
                [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": 0.2}],
                [],
            ]
        )


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_basic(backend):
    assert_backend = functools.partial(_assert, backend=backend)

    array = vector.Array(
        [[{"x": 1, "y": 2}], [], [{"x": 3, "y": 4}]],
        backend=backend,
    )
    assert isinstance(array, vector.backends.awkward.VectorArray2D)
    assert_backend(array.x, [[1], [], [3]])
    assert_backend(array.y, [[2], [], [4]])
    assert_backend(array.rho, [[2.23606797749979], [], [5]])

    assert isinstance(array[2, 0], vector.backends.awkward.VectorRecord2D)
    if backend == "cpu":
        (a,), (), (c,) = array.phi.tolist()
        assert a == pytest.approx(1.1071487177940904)
        assert c == pytest.approx(0.9272952180016122)
        assert array[2, 0].rho == 5
        assert array.deltaphi(array).tolist() == [[0], [], [0]]
    else:
        # at least make sure they run
        _ = array.phi
        _ = array[2, 0].rho
        _ = array.deltaphi(array)

    array = vector.Array(
        [[{"pt": 1, "phi": 2}], [], [{"pt": 3, "phi": 4}]],
        backend=backend,
    )
    assert isinstance(array, vector.backends.awkward.MomentumArray2D)
    assert_backend(array.pt, [[1], [], [3]])

    array = vector.Array(
        [
            [{"x": 1, "y": 2, "z": 3, "wow": 99}],
            [],
            [{"x": 4, "y": 5, "z": 6, "wow": 123}],
        ],
        backend=backend,
    )
    assert isinstance(array, vector.backends.awkward.VectorArray3D)
    assert_backend(array.wow, [[99], [], [123]])


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_rotateZ(backend):
    assert_backend = functools.partial(_assert, backend=backend)

    array = vector.Array(
        [[{"pt": 1, "phi": 0}], [], [{"pt": 2, "phi": 1}]],
        backend=backend,
    )
    out = array.rotateZ(1)
    assert isinstance(out, vector.backends.awkward.MomentumArray2D)
    assert_backend(out, [[{"rho": 1, "phi": 1}], [], [{"rho": 2, "phi": 2}]])

    array = vector.Array(
        [[{"x": 1, "y": 0, "wow": 99}], [], [{"x": 2, "y": 1, "wow": 123}]],
        backend=backend,
    )
    out = array.rotateZ(0.1)
    assert isinstance(out, vector.backends.awkward.VectorArray2D)
    assert_backend(out.wow, [[99], [], [123]])


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_projection(backend):
    assert_backend = functools.partial(_assert, backend=backend)

    array = vector.Array(
        [
            [{"x": 1, "y": 2, "z": 3, "wow": 99}],
            [],
            [{"x": 4, "y": 5, "z": 6, "wow": 123}],
        ],
        backend=backend,
    )
    out = array.to_Vector2D()
    assert isinstance(out, vector.backends.awkward.VectorArray2D)
    assert_backend(
        out,
        [
            [{"x": 1, "y": 2, "wow": 99}],
            [],
            [{"x": 4, "y": 5, "wow": 123}],
        ],
    )

    out = array.to_Vector4D()
    assert isinstance(out, vector.backends.awkward.VectorArray4D)
    assert_backend(
        out,
        [
            [{"x": 1, "y": 2, "z": 3, "t": 0, "wow": 99}],
            [],
            [{"x": 4, "y": 5, "z": 6, "t": 0, "wow": 123}],
        ],
    )

    out = array.to_rhophietatau()
    assert isinstance(out, vector.backends.awkward.VectorArray4D)
    if backend == "cpu":
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


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_add(backend):
    assert_backend = functools.partial(_assert, backend=backend)

    one = vector.Array(
        [[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}], [], [{"x": 3, "y": 3.3}]],
        backend=backend,
    )

    two = vector.Array(
        [{"x": 10, "y": 20}, {"x": 100, "y": 200}, {"x": 1000, "y": 2000}],
        backend=backend,
    )
    assert isinstance(one.add(two), vector.backends.awkward.VectorArray2D)
    assert isinstance(two.add(one), vector.backends.awkward.VectorArray2D)
    assert_backend(
        one.add(two),
        [
            [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
            [],
            [{"x": 1003, "y": 2003.3}],
        ],
    )
    assert_backend(
        two.add(one),
        [
            [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
            [],
            [{"x": 1003, "y": 2003.3}],
        ],
    )

    two = vector.array({"x": [10, 100, 1000], "y": [20, 200, 2000]})
    assert isinstance(one.add(two), vector.backends.awkward.VectorArray2D)
    assert isinstance(two.add(one), vector.backends.awkward.VectorArray2D)
    assert_backend(
        one.add(two),
        [
            [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
            [],
            [{"x": 1003, "y": 2003.3}],
        ],
    )
    assert_backend(
        two.add(one),
        [
            [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
            [],
            [{"x": 1003, "y": 2003.3}],
        ],
    )

    two = vector.obj(x=10, y=20)
    assert isinstance(one.add(two), vector.backends.awkward.VectorArray2D)
    assert isinstance(two.add(one), vector.backends.awkward.VectorArray2D)
    assert_backend(
        one.add(two),
        [
            [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
            [],
            [{"x": 13, "y": 23.3}],
        ],
    )
    assert_backend(
        two.add(one),
        [
            [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
            [],
            [{"x": 13, "y": 23.3}],
        ],
    )


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_ufuncs(backend):
    assert_backend = functools.partial(_assert, backend=backend)

    array = vector.Array(
        [[{"x": 3, "y": 4}, {"x": 5, "y": 12}], [], [{"x": 8, "y": 15}]],
        backend=backend,
    )
    assert_backend(abs(array), [[5, 13], [], [17]])
    assert_backend((array**2), [[5**2, 13**2], [], [17**2]])

    one = vector.Array(
        [[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}], [], [{"x": 3, "y": 3.3}]],
        backend=backend,
    )
    two = vector.Array(
        [{"x": 10, "y": 20}, {"x": 100, "y": 200}, {"x": 1000, "y": 2000}],
        backend=backend,
    )
    assert_backend(
        one + two,
        [
            [{"x": 11, "y": 21.1}, {"x": 12, "y": 22.2}],
            [],
            [{"x": 1003, "y": 2003.3}],
        ],
    )

    assert_backend(
        one * 10,
        [
            [{"x": 10, "y": 11}, {"x": 20, "y": 22}],
            [],
            [{"x": 30, "y": 33}],
        ],
    )


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_zip(backend):
    assert_backend = functools.partial(_assert, backend=backend)

    v = ak.to_backend(vector.zip({"x": [[], [1]], "y": [[], [1]]}), backend=backend)
    assert isinstance(v, vector.backends.awkward.VectorArray2D)
    assert isinstance(v[1], vector.backends.awkward.VectorArray2D)
    assert isinstance(v[1, 0], vector.backends.awkward.VectorRecord2D)
    assert_backend(v, [[], [{"x": 1, "y": 1}]])
    assert_backend(v.x, [[], [1]])
    assert_backend(v.y, [[], [1]])


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers were added before awkward==2.2.3, but had some bugs",
)
@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_sum_2d(backend):
    assert_backend = functools.partial(_assert, backend=backend)

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
        ],
        backend=backend,
    )
    # typetracer backend does not implement ak.almost_equal
    if backend != "typetracer":
        assert_backend(
            ak.almost_equal(
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
            ),
            True,
        )
        assert_backend(
            ak.almost_equal(
                ak.sum(v, axis=0, keepdims=False),
                vector.Array(
                    [
                        {"x": 1.950340654403632, "y": 0.3953536233081677},
                        {"x": 7.604510287376507, "y": 2.3523506924148467},
                        {"x": 0.9950041652780258, "y": 0.09983341664682815},
                    ]
                ),
            ),
            True,
        )
        assert_backend(
            ak.almost_equal(
                ak.sum(v, axis=1, keepdims=True),
                ak.to_regular(
                    vector.Array(
                        [
                            [{"x": 4.915270, "y": 0.89451074}],
                            [{"x": 5.63458463, "y": 1.95302699}],
                        ]
                    )
                ),
            ),
            True,
        )
        assert_backend(
            ak.almost_equal(
                ak.sum(v, axis=1, keepdims=False),
                vector.Array(
                    [
                        {"x": 4.915270, "y": 0.89451074},
                        {"x": 5.63458463, "y": 1.95302699},
                    ]
                ),
            ),
            True,
        )
        assert_backend(
            ak.almost_equal(
                ak.sum(v.mask[[False, True]], axis=1),
                vector.Array(
                    [
                        {"x": 5.63458463, "y": 1.95302699},
                    ]
                )[[None, 0]],
            ),
            True,
        )


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers were added before awkward==2.2.3, but had some bugs",
)
@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_sum_3d(backend):
    assert_backend = functools.partial(_assert, backend=backend)

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
        ],
        backend=backend,
    )
    assert_backend(
        ak.sum(v, axis=0, keepdims=True),
        [
            [
                {"x": 2, "y": 4, "z": 6},
                {"x": 8, "y": 10, "z": 12},
                {"x": 1, "y": 1, "z": 1},
            ]
        ],
    )
    assert_backend(
        ak.sum(v, axis=0, keepdims=False),
        [
            {"x": 2, "y": 4, "z": 6},
            {"x": 8, "y": 10, "z": 12},
            {"x": 1, "y": 1, "z": 1},
        ],
    )
    assert_backend(
        ak.sum(v, axis=1, keepdims=True),
        [
            [{"x": 5, "y": 7, "z": 9}],
            [{"x": 6, "y": 8, "z": 10}],
        ],
    )
    assert_backend(
        ak.sum(v, axis=1, keepdims=False),
        [
            {"x": 5, "y": 7, "z": 9},
            {"x": 6, "y": 8, "z": 10},
        ],
    )
    assert_backend(
        ak.sum(v.mask[[False, True]], axis=1),
        [
            None,
            {"x": 6, "y": 8, "z": 10},
        ],
    )


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers were added before awkward==2.2.3, but had some bugs",
)
@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_sum_4d(backend):
    assert_backend = functools.partial(_assert, backend=backend)

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
        ],
        backend=backend,
    )
    assert_backend(
        ak.sum(v, axis=0, keepdims=True),
        [
            [
                {"t": 12, "z": 6, "x": 2, "y": 4},
                {"t": 2, "z": 12, "x": 8, "y": 10},
                {"t": 3, "z": 1, "x": 1, "y": 1},
            ]
        ],
    )
    assert_backend(
        ak.sum(v, axis=0, keepdims=False),
        [
            {"t": 12, "z": 6, "x": 2, "y": 4},
            {"t": 2, "z": 12, "x": 8, "y": 10},
            {"t": 3, "z": 1, "x": 1, "y": 1},
        ],
    )
    assert_backend(
        ak.sum(v, axis=1, keepdims=True),
        [
            [{"t": 9, "z": 9, "x": 5, "y": 7}],
            [{"t": 8, "z": 10, "x": 6, "y": 8}],
        ],
    )
    assert_backend(
        ak.sum(v, axis=1, keepdims=False),
        [
            {"t": 9, "z": 9, "x": 5, "y": 7},
            {"t": 8, "z": 10, "x": 6, "y": 8},
        ],
    )
    assert_backend(
        ak.sum(v.mask[[False, True]], axis=1),
        [
            None,
            {"t": 8, "z": 10, "x": 6, "y": 8},
        ],
    )


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers were added before awkward==2.2.3, but had some bugs",
)
@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_count_nonzero_2d(backend):
    assert_backend = functools.partial(_assert, backend=backend)

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
        ],
        backend=backend,
    )
    assert_backend(ak.count_nonzero(v, axis=1), [2, 3])
    assert_backend(ak.count_nonzero(v, axis=1, keepdims=True), [[2], [3]])
    assert_backend(ak.count_nonzero(v, axis=0), [2, 2, 1])
    assert_backend(ak.count_nonzero(v, axis=0, keepdims=True), [[2, 2, 1]])


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers were added before awkward==2.2.3, but had some bugs",
)
@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_count_nonzero_3d(backend):
    assert_backend = functools.partial(_assert, backend=backend)

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
        ],
        backend=backend,
    )
    assert_backend(ak.count_nonzero(v, axis=1), [2, 3])
    assert_backend(ak.count_nonzero(v, axis=1, keepdims=True), [[2], [3]])
    assert_backend(ak.count_nonzero(v, axis=0), [2, 2, 1])
    assert_backend(ak.count_nonzero(v, axis=0, keepdims=True), [[2, 2, 1]])


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers were added before awkward==2.2.3, but had some bugs",
)
@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_count_nonzero_4d(backend):
    assert_backend = functools.partial(_assert, backend=backend)

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
        ],
        backend=backend,
    )
    assert_backend(ak.count_nonzero(v, axis=1), [3, 3])
    assert_backend(ak.count_nonzero(v, axis=1, keepdims=True), [[3], [3]])
    assert_backend(ak.count_nonzero(v, axis=0), [2, 2, 2])
    assert_backend(ak.count_nonzero(v, axis=0, keepdims=True), [[2, 2, 2]])

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
        ],
        backend=backend,
    )
    assert_backend(ak.count_nonzero(v2, axis=1), [3, 2])
    assert_backend(ak.count_nonzero(v2, axis=1, keepdims=True), [[3], [2]])
    assert_backend(ak.count_nonzero(v2, axis=0), [2, 2, 1])
    assert_backend(ak.count_nonzero(v2, axis=0, keepdims=True), [[2, 2, 1]])


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers were added before awkward==2.2.3, but had some bugs",
)
@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_count_2d(backend):
    assert_backend = functools.partial(_assert, backend=backend)

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
        ],
        backend=backend,
    )
    assert_backend(ak.count(v, axis=1), [3, 3])
    assert_backend(ak.count(v, axis=1, keepdims=True), [[3], [3]])
    assert_backend(ak.count(v, axis=0), [2, 2, 2])
    assert_backend(ak.count(v, axis=0, keepdims=True), [[2, 2, 2]])
    assert_backend(ak.count(v.mask[[False, True]], axis=1), [None, 3])


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers were added before awkward==2.2.3, but had some bugs",
)
@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_count_3d(backend):
    assert_backend = functools.partial(_assert, backend=backend)

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
    assert_backend(ak.count(v, axis=1), [3, 3])
    assert_backend(ak.count(v, axis=1, keepdims=True), [[3], [3]])
    assert_backend(ak.count(v, axis=0), [2, 2, 2])
    assert_backend(ak.count(v, axis=0, keepdims=True), [[2, 2, 2]])
    assert_backend(ak.count(v.mask[[False, True]], axis=1), [None, 3])


@pytest.mark.skipif(
    awkward_without_record_reducers,
    reason="record reducers were added before awkward==2.2.3, but had some bugs",
)
@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_count_4d(backend):
    assert_backend = functools.partial(_assert, backend=backend)

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
    assert_backend(ak.count(v, axis=1), [3, 3])
    assert_backend(ak.count(v, axis=1, keepdims=True), [[3], [3]])
    assert_backend(ak.count(v, axis=0), [2, 2, 2])
    assert_backend(ak.count(v, axis=0, keepdims=True), [[2, 2, 2]])
    assert_backend(ak.count(v.mask[[False, True]], axis=1), [None, 3])


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_like(backend):
    assert_backend = functools.partial(_assert, backend=backend)

    v1 = ak.to_backend(
        vector.zip(
            {
                "x": [10.0, 20.0, 30.0],
                "y": [-10.0, 20.0, 30.0],
            },
        ),
        backend=backend,
    )
    v2 = ak.to_backend(
        vector.zip(
            {
                "x": [10.0, 20.0, 30.0],
                "y": [-10.0, 20.0, 30.0],
                "z": [5.0, 1.0, 1.0],
            },
        ),
        backend=backend,
    )
    v3 = ak.to_backend(
        vector.zip(
            {
                "x": [10.0, 20.0, 30.0],
                "y": [-10.0, 20.0, 30.0],
                "z": [5.0, 1.0, 1.0],
                "t": [16.0, 31.0, 46.0],
            },
        ),
        backend=backend,
    )

    v1_v2 = ak.to_backend(
        vector.zip(
            {
                "x": [20.0, 40.0, 60.0],
                "y": [-20.0, 40.0, 60.0],
            },
        ),
        backend=backend,
    )
    v2_v1 = ak.to_backend(
        vector.zip(
            {
                "x": [20.0, 40.0, 60.0],
                "y": [-20.0, 40.0, 60.0],
                "z": [5.0, 1.0, 1.0],
            },
        ),
        backend=backend,
    )
    v2_v3 = ak.to_backend(
        vector.zip(
            {
                "x": [20.0, 40.0, 60.0],
                "y": [-20.0, 40.0, 60.0],
                "z": [10.0, 2.0, 2.0],
            },
        ),
        backend=backend,
    )
    v3_v2 = ak.to_backend(
        vector.zip(
            {
                "x": [20.0, 40.0, 60.0],
                "y": [-20.0, 40.0, 60.0],
                "z": [10.0, 2.0, 2.0],
                "t": [16.0, 31.0, 46.0],
            },
        ),
        backend=backend,
    )
    v1_v3 = ak.to_backend(
        vector.zip(
            {
                "x": [20.0, 40.0, 60.0],
                "y": [-20.0, 40.0, 60.0],
                "z": [5.0, 1.0, 1.0],
                "t": [16.0, 31.0, 46.0],
            },
        ),
        backend=backend,
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
    assert_backend(v1 + v2.like(v1), v1_v2)
    assert_backend(v2.like(v1) + v1, v1_v2)
    # 2D + 4D.like(2D) = 2D
    assert_backend(v1 + v3.like(v1), v1_v2)
    assert_backend(v3.like(v1) + v1, v1_v2)
    # 3D + 2D.like(3D) = 3D
    assert_backend(v2 + v1.like(v2), v2_v1)
    assert_backend(v1.like(v2) + v2, v2_v1)
    # 3D + 4D.like(3D) = 3D
    assert_backend(v2 + v3.like(v2), v2_v3)
    assert_backend(v3.like(v2) + v2, v2_v3)
    # 4D + 2D.like(4D) = 4D
    assert_backend(v3 + v1.like(v3), v1_v3)
    assert_backend(v1.like(v3) + v3, v1_v3)
    # 4D + 3D.like(4D) = 4D
    assert_backend(v3 + v2.like(v3), v3_v2)
    assert_backend(v2.like(v3) + v3, v3_v2)

    v1 = ak.to_backend(
        vector.zip(
            {
                "px": [10.0, 20.0, 30.0],
                "py": [-10.0, 20.0, 30.0],
            },
        ),
        backend=backend,
    )
    v2 = ak.to_backend(
        vector.zip(
            {
                "px": [10.0, 20.0, 30.0],
                "py": [-10.0, 20.0, 30.0],
                "pz": [5.0, 1.0, 1.0],
            },
        ),
        backend=backend,
    )
    v3 = ak.to_backend(
        vector.zip(
            {
                "px": [10.0, 20.0, 30.0],
                "py": [-10.0, 20.0, 30.0],
                "pz": [5.0, 1.0, 1.0],
                "t": [16.0, 31.0, 46.0],
            },
        ),
        backend=backend,
    )

    # 2D + 3D.like(2D) = 2D
    assert_backend(v1 + v2.like(v1), v1_v2)
    assert_backend(v2.like(v1) + v1, v1_v2)
    # 2D + 4D.like(2D) = 2D
    assert_backend(v1 + v3.like(v1), v1_v2)
    assert_backend(v3.like(v1) + v1, v1_v2)
    # 3D + 2D.like(3D) = 3D
    assert_backend(v2 + v1.like(v2), v2_v1)
    assert_backend(v1.like(v2) + v2, v2_v1)
    # 3D + 4D.like(3D) = 3D
    assert_backend(v2 + v3.like(v2), v2_v3)
    assert_backend(v3.like(v2) + v2, v2_v3)
    # 4D + 2D.like(4D) = 4D
    assert_backend(v3 + v1.like(v3), v1_v3)
    assert_backend(v1.like(v3) + v3, v1_v3)
    # 4D + 3D.like(4D) = 4D
    assert_backend(v3 + v2.like(v3), v3_v2)
    assert_backend(v2.like(v3) + v3, v3_v2)


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_handler_of(backend):
    assert_backend = functools.partial(_assert, backend=backend)

    numpy_vec = ak.to_backend(
        vector.array(
            {
                "x": [1.1, 1.2, 1.3, 1.4, 1.5],
                "y": [2.1, 2.2, 2.3, 2.4, 2.5],
                "z": [3.1, 3.2, 3.3, 3.4, 3.5],
            }
        ),
        backend=backend,
    )
    awkward_vec2 = ak.to_backend(
        vector.zip(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [-1.0, 2.0, 3.0],
                "z": [5.0, 10.0, 15.0],
                "t": [16.0, 31.0, 46.0],
            },
        ),
        backend=backend,
    )
    object_vec = VectorObject2D.from_xy(1.0, 1.0)
    protocol = vector._methods._handler_of(object_vec, numpy_vec, awkward_vec2)
    # chooses awkward backend
    assert_backend(protocol, awkward_vec2)


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_momentum_coordinate_transforms(backend):
    awkward_vec = ak.to_backend(
        vector.zip(
            {
                "px": [1.0, 2.0, 3.0],
                "py": [-1.0, 2.0, 3.0],
            },
        ),
        backend=backend,
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


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_momentum_preservation(backend):
    v1 = ak.to_backend(
        vector.zip(
            {
                "px": [10.0, 20.0, 30.0],
                "py": [-10.0, 20.0, 30.0],
            },
        ),
        backend=backend,
    )
    v2 = ak.to_backend(
        vector.zip(
            {
                "x": [10.0, 20.0, 30.0],
                "y": [-10.0, 20.0, 30.0],
                "z": [5.0, 1.0, 1.0],
            },
        ),
        backend=backend,
    )

    v3 = ak.to_backend(
        vector.zip(
            {
                "px": [10.0, 20.0, 30.0],
                "py": [-10.0, 20.0, 30.0],
                "pz": [5.0, 1.0, 1.0],
                "t": [16.0, 31.0, 46.0],
            },
        ),
        backend=backend,
    )

    # momentum + generic = momentum
    # 2D + 3D.like(2D) = 2D
    assert isinstance(v1 + v2.like(v1), vector.backends.awkward.MomentumAwkward2D)
    assert isinstance(v2.like(v1) + v1, vector.backends.awkward.MomentumAwkward2D)
    # 2D + 4D.like(2D) = 2D
    assert isinstance(v1 + v3.like(v1), vector.backends.awkward.MomentumAwkward2D)
    assert isinstance(v3.like(v1) + v1, vector.backends.awkward.MomentumAwkward2D)
    # 3D + 2D.like(3D) = 3D
    assert isinstance(v2 + v1.like(v2), vector.backends.awkward.MomentumAwkward3D)
    assert isinstance(v1.like(v2) + v2, vector.backends.awkward.MomentumAwkward3D)
    # 3D + 4D.like(3D) = 3D
    assert isinstance(v2 + v3.like(v2), vector.backends.awkward.MomentumAwkward3D)
    assert isinstance(v3.like(v2) + v2, vector.backends.awkward.MomentumAwkward3D)
    # 4D + 2D.like(4D) = 4D
    assert isinstance(v3 + v1.like(v3), vector.backends.awkward.MomentumAwkward4D)
    assert isinstance(v1.like(v3) + v3, vector.backends.awkward.MomentumAwkward4D)
    # 4D + 3D.like(4D) = 4D
    assert isinstance(v3 + v2.like(v3), vector.backends.awkward.MomentumAwkward4D)
    assert isinstance(v2.like(v3) + v3, vector.backends.awkward.MomentumAwkward4D)


@pytest.mark.parametrize("backend", ["cpu", "typetracer"])
def test_subclass_fields(backend):
    @ak.mixin_class(vector.backends.awkward.behavior)
    class TwoVector(vector.backends.awkward.MomentumAwkward2D):
        pass

    @ak.mixin_class(vector.backends.awkward.behavior)
    class ThreeVector(vector.backends.awkward.MomentumAwkward3D):
        pass

    @ak.mixin_class(vector.backends.awkward.behavior)
    class LorentzVector(vector.backends.awkward.MomentumAwkward4D):
        @ak.mixin_class_method(np.divide, {numbers.Number})
        def divide(self, factor):
            return self.scale(1 / factor)

    LorentzVectorArray.ProjectionClass2D = TwoVectorArray  # noqa: F821
    LorentzVectorArray.ProjectionClass3D = ThreeVectorArray  # noqa: F821
    LorentzVectorArray.ProjectionClass4D = LorentzVectorArray  # noqa: F821
    LorentzVectorArray.MomentumClass = LorentzVectorArray  # noqa: F821

    vec = ak.to_backend(
        ak.zip(
            {
                "pt": [[1, 2], [], [3], [4]],
                "eta": [[1.2, 1.4], [], [1.6], [3.4]],
                "phi": [[0.3, 0.4], [], [0.5], [0.6]],
                "energy": [[50, 51], [], [52], [60]],
            },
            with_name="LorentzVector",
            behavior=vector.backends.awkward.behavior,
        ),
        backend=backend,
    )

    assert vec.like(vector.obj(x=1, y=2)).fields == ["rho", "phi"]
    assert vec.like(vector.obj(x=1, y=2, z=3)).fields == ["rho", "phi", "eta"]
    assert (vec / 2).fields == ["rho", "phi", "eta", "t"]
