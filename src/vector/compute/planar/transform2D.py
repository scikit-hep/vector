# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.compute.planar import x, y
from vector.methods import AzimuthalRhoPhi, AzimuthalXY, _aztype

# Rotation is only computed in Cartesian coordinates; the rest are conversions.


def cartesian(lib, xx, xy, yx, yy, x, y):
    return (xx * x + xy * y, yx * x + yy * y)


def rhophi(lib, xx, xy, yx, yy, rho, phi):
    return cartesian(lib, xx, xy, yx, yy, x.rhophi(rho, phi), y.rhophi(rho, phi))


dispatch_map = {
    (AzimuthalXY,): (cartesian, AzimuthalXY),
    (AzimuthalRhoPhi,): (rhophi, AzimuthalXY),
}


def dispatch(obj, v):
    function, *returns = dispatch_map[
        _aztype(v),
    ]
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            function(
                v.lib, obj["xx"], obj["xy"], obj["yx"], obj["yy"], *v.azimuthal.elements
            ),
            returns,
        )
