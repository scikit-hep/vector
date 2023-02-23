# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import vector


class CustomVector(vector.VectorObject4D):
    pass


def test_handler_of():
    object_a = CustomVector.from_xyzt(0.0, 0.0, 0.0, 0.0)
    object_b = CustomVector.from_xyzt(1.0, 1.0, 1.0, 1.0)
    protocol = vector._methods._handler_of(object_a, object_b)
    assert protocol == object_a
