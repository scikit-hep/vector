# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import sys
import typing

if sys.version_info < (3, 8):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


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
