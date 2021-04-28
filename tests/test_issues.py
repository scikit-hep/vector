# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import pytest

import vector


def test_issue_99():
    ak = pytest.importorskip("awkward")
    vector.register_awkward()
    vec = ak.Array([{"x": 1.0, "y": 2.0, "z": 3.0}], with_name="Vector3D")
    assert vec.to_xyz().tolist() == [{"x": 1.0, "y": 2.0, "z": 3.0}]
    assert vec[0].to_xyz().tolist() == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert vec[0].to_rhophiz().tolist() == {
        "rho": 2.23606797749979,
        "phi": 1.1071487177940904,
        "z": 3.0,
    }
