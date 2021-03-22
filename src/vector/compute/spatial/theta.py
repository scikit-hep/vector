# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.compute.spatial import costheta
from vector.methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    _aztype,
    _from_signature,
    _ltype,
)


def xy_z(lib, x, y, z):
    return lib.arccos(costheta.xy_z(lib, x, y, z))


def xy_theta(lib, x, y, theta):
    return theta


def xy_eta(lib, x, y, eta):
    return 2.0 * lib.arctan(lib.exp(-eta))


def rhophi_z(lib, rho, phi, z):
    return lib.arccos(costheta.rhophi_z(lib, rho, phi, z))


def rhophi_theta(lib, rho, phi, theta):
    return theta


def rhophi_eta(lib, rho, phi, eta):
    return 2.0 * lib.arctan(lib.exp(-eta))


dispatch_map = {
    (AzimuthalXY, LongitudinalZ): (xy_z, float),
    (AzimuthalXY, LongitudinalTheta): (xy_theta, float),
    (AzimuthalXY, LongitudinalEta): (xy_eta, float),
    (AzimuthalRhoPhi, LongitudinalZ): (rhophi_z, float),
    (AzimuthalRhoPhi, LongitudinalTheta): (rhophi_theta, float),
    (AzimuthalRhoPhi, LongitudinalEta): (rhophi_eta, float),
}


def dispatch(v):
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v),
            _ltype(v),
        ),
    )
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            type(v), function(v.lib, *v.azimuthal.elements, *v.longitudinal.elements), returns
        )
