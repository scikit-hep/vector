# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import typing
from typing import Protocol, TypedDict

__all__ = [
    "BoolCollection",
    "FloatArray",
    "Protocol",
    "ScalarCollection",
    "TransformProtocol2D",
    "TransformProtocol3D",
    "TransformProtocol4D",
]


def __dir__() -> list[str]:
    return __all__


# Represents a number, a NumPy array, an Awkward Array, etc., of non-vectors.
ScalarCollection = typing.Any

# Represents a bool, a NumPy array of bools, an Awkward Array of bools, etc.
BoolCollection = typing.Any


class TransformProtocol2D(TypedDict):
    xx: ScalarCollection
    xy: ScalarCollection
    yx: ScalarCollection
    yy: ScalarCollection


class TransformProtocol3D(TypedDict):
    xx: ScalarCollection
    xy: ScalarCollection
    xz: ScalarCollection
    yx: ScalarCollection
    yy: ScalarCollection
    yz: ScalarCollection
    zx: ScalarCollection
    zy: ScalarCollection
    zz: ScalarCollection


class TransformProtocol4D(TypedDict):
    xx: ScalarCollection
    xy: ScalarCollection
    xz: ScalarCollection
    xt: ScalarCollection
    yx: ScalarCollection
    yy: ScalarCollection
    yz: ScalarCollection
    yt: ScalarCollection
    zx: ScalarCollection
    zy: ScalarCollection
    zz: ScalarCollection
    zt: ScalarCollection
    tx: ScalarCollection
    ty: ScalarCollection
    tz: ScalarCollection
    tt: ScalarCollection


if typing.TYPE_CHECKING:
    import numpy.typing

    FloatArray = numpy.typing.NDArray[numpy.float64]
else:
    import numpy

    FloatArray = numpy.ndarray
