# -*- coding: utf-8 -*-
import awkward as ak
import pytest

numba = pytest.importorskip("numba")

from vector.numba.awkward.lorentz.xyzt import behavior  # noqa: E402


def ak_vector(x, y, z, t):
    return ak.Array(
        [{"x": x, "y": y, "z": z, "t": t}], with_name="LorentzXYZT", behavior=behavior
    )


def test_add_vector():
    v1 = ak_vector(1.0, 2.0, 3.0, 4.0)
    v2 = ak_vector(4.0, 3.0, 2.0, 1.0)

    @numba.njit
    def test_addition(v1, v2):
        return v1[0] + v2[0]

    assert repr(test_addition(v1, v2)) == repr(v1[0] + v2[0])


# def test_add_scalar():
#     v1 = ak_vector(1.0, 2.0, 3.0, 4.0)
#
#     @numba.njit
#     def test_addition_scalar(v):
#         return (
#             v[0] + 1.0,
#             1.0 + v[0],
#         )
#
#     v12, v21 = test_addition_scalar(v1)
#
#     ans = v1[0] + 1
#     assert ans == v12
#     assert ans == v21
