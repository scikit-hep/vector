# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.compute.planar import rho
from vector.methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    _aztype,
    _flavor_of,
    _from_signature,
)


def xy(lib, x, y):
    norm = rho.xy(lib, x, y)
    return (lib.nan_to_num(x / norm, nan=0), lib.nan_to_num(y / norm, nan=0))


def rhophi(lib, rho, phi):
    return (1, phi)


dispatch_map = {
    (AzimuthalXY,): (xy, AzimuthalXY),
    (AzimuthalRhoPhi,): (rhophi, AzimuthalRhoPhi),
}


def dispatch(v):
    function, *returns = _from_signature(__name__, dispatch_map, (_aztype(v),))
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            _flavor_of(v), function(v.lib, *v.azimuthal.elements), returns
        )
