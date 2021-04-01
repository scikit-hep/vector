# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

"""
.. code-block:: python

    Spatial.unit(self)
"""

import numpy

from vector._compute.spatial import mag
from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    _aztype,
    _flavor_of,
    _from_signature,
    _ltype,
)


def xy_z(lib, x, y, z):
    norm = mag.xy_z(lib, x, y, z)
    return (
        lib.nan_to_num(x / norm, nan=0),
        lib.nan_to_num(y / norm, nan=0),
        lib.nan_to_num(z / norm, nan=0),
    )


def xy_theta(lib, x, y, theta):
    norm = mag.xy_theta(lib, x, y, theta)
    return (lib.nan_to_num(x / norm, nan=0), lib.nan_to_num(y / norm, nan=0), theta)


def xy_eta(lib, x, y, eta):
    norm = mag.xy_eta(lib, x, y, eta)
    return (lib.nan_to_num(x / norm, nan=0), lib.nan_to_num(y / norm, nan=0), eta)


def rhophi_z(lib, rho, phi, z):
    norm = mag.rhophi_z(lib, rho, phi, z)
    return (lib.nan_to_num(rho / norm, nan=0), phi, lib.nan_to_num(z / norm, nan=0))


def rhophi_theta(lib, rho, phi, theta):
    norm = mag.rhophi_theta(lib, rho, phi, theta)
    return (lib.nan_to_num(rho / norm, nan=0), phi, theta)


def rhophi_eta(lib, rho, phi, eta):
    norm = mag.rhophi_eta(lib, rho, phi, eta)
    return (lib.nan_to_num(rho / norm, nan=0), phi, eta)


dispatch_map = {
    (AzimuthalXY, LongitudinalZ): (xy_z, AzimuthalXY, LongitudinalZ),
    (AzimuthalXY, LongitudinalTheta): (xy_theta, AzimuthalXY, LongitudinalTheta),
    (AzimuthalXY, LongitudinalEta): (xy_eta, AzimuthalXY, LongitudinalEta),
    (AzimuthalRhoPhi, LongitudinalZ): (rhophi_z, AzimuthalRhoPhi, LongitudinalZ),
    (AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_theta,
        AzimuthalRhoPhi,
        LongitudinalTheta,
    ),
    (AzimuthalRhoPhi, LongitudinalEta): (rhophi_eta, AzimuthalRhoPhi, LongitudinalEta),
}


def dispatch(v: typing.Any) -> typing.Any:
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
            _flavor_of(v),
            function(v.lib, *v.azimuthal.elements, *v.longitudinal.elements),
            returns,
            1,
        )
