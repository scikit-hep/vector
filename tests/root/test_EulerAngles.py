# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest

# from hypothesis import given
# from hypothesis import strategies as st

# import vector

# If ROOT is not available, skip these tests.
ROOT = pytest.importorskip("ROOT")

# 4D constructor arguments to get all the weird cases.
constructor = [
    (0, 0, 0, 0),
    (0, 0, 1, 0),  # theta == 0.0
    (0, 0, -1, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 4294967296),
    (0, 4294967296, 0, 0),
    (0, 0, 0, 10),
    (0, 0, 0, -10),
    (1, 2, 3, 0),
    (1, 2, 3, 10),
    (1, 2, 3, -10),
    (1.0, 2.0, 3.0, 2.5),
    (1, 2, 3, 2.5),
    (1, 2, 3, -2.5),
]

# Coordinate conversion methods to apply to the VectorObject4D.
coordinate_list = [
    "to_xyzt",
    "to_xythetat",  # may fail for constructor2
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
