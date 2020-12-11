# -*- coding: utf-8 -*-
from collections import namedtuple

import numpy as np
from numpy.testing import assert_almost_equal as _aae

import pytest

from vector.numpy.lorentz.xyzt import LorentzXYZT
from vector.single.lorentz.xyzt import LorentzXYZTFree

try:
    from vector.awkward.lorentz.xyzt import behavior
except ImportError:
    behavior = None


# Wrapped assert because Awkward arrays can't be used directly here
def assert_almost_equal(actual, expected):
    _aae(np.asarray(actual), np.asarray(expected))


# Open for future expansion
VectorInfo = namedtuple("VectorInfo", ("init",))


params = {
    "Free": VectorInfo(lambda x, y, z, t: LorentzXYZTFree(x, y, z, t)),
    "NumPyScalar": VectorInfo(lambda x, y, z, t: LorentzXYZT(x, y, z, t)),
    "NumPyArray": VectorInfo(lambda x, y, z, t: LorentzXYZT([x], [y], z, t)),
}

if behavior:
    import awkward as ak

    params["Awkward"] = VectorInfo(
        lambda x, y, z, t: ak.Array(
            [{"x": x, "y": y, "z": z, "t": t}],
            with_name="LorentzXYZT",
            behavior=behavior,
        )
    )


@pytest.fixture(params=params.values(), ids=params.keys())
def vi(request):
    cls = request.param
    return cls


def test_mag(vi):
    v = vi.init(0, 3, 4, 5)
    assert_almost_equal(0, v.mag)


def test_add_vector(vi):
    v1 = vi.init(1, 2, 3, 4)
    v2 = vi.init(4, 3, 2, 1)

    v12 = v1 + v2

    assert_almost_equal(5, v12.x)
    assert_almost_equal(5, v12.y)
    assert_almost_equal(5, v12.z)
    assert_almost_equal(5, v12.t)


def test_add_scalar(vi):
    v1 = vi.init(1, 2, 3, 4)

    v12 = v1 + 1

    assert_almost_equal(2, v12.x)
    assert_almost_equal(3, v12.y)
    assert_almost_equal(4, v12.z)
    assert_almost_equal(5, v12.t)

    v21 = 1 + v1

    assert_almost_equal(2, v21.x)
    assert_almost_equal(3, v21.y)
    assert_almost_equal(4, v21.z)
    assert_almost_equal(5, v21.t)


def test_basic_scalar(vi):
    v1 = vi.init(1, 2, 3, 4)
    v12 = v1 * 2
    assert_almost_equal(v1.x * 2, v12.x)
    assert_almost_equal(v1.y * 2, v12.y)
    assert_almost_equal(v1.z * 2, v12.z)
    assert_almost_equal(v1.t * 2, v12.t)

    v21 = 2 * v1
    assert_almost_equal(v1.x * 2, v21.x)
    assert_almost_equal(v1.y * 2, v21.y)
    assert_almost_equal(v1.z * 2, v21.z)
    assert_almost_equal(v1.t * 2, v21.t)


def test_dot(vi):
    v1 = vi.init(1, 2, 3, 4)
    v2 = vi.init(0, 2, 2, 5)

    res = v1.dot(v2)

    assert_almost_equal(20 - 6 - 4, res)


def test_mul_vec(vi):
    v1 = vi.init(1, 2, 3, 4)
    v2 = vi.init(0, 2, 2, 5)

    res = v1 * v2

    assert_almost_equal(20 - 6 - 4, res)


def test_mul_scalar(vi):
    v1 = vi.init(1, 2, 3, 4)
    v12 = v1 * 2
    assert_almost_equal(v1.x * 2, v12.x)
    assert_almost_equal(v1.y * 2, v12.y)
    assert_almost_equal(v1.z * 2, v12.z)
    assert_almost_equal(v1.t * 2, v12.t)

    v21 = 2 * v1
    assert_almost_equal(v1.x * 2, v21.x)
    assert_almost_equal(v1.y * 2, v21.y)
    assert_almost_equal(v1.z * 2, v21.z)
    assert_almost_equal(v1.t * 2, v21.t)
