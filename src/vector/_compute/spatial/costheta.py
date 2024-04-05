# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    @property
    Spatial.costheta(self)
"""

from __future__ import annotations

import typing
from math import inf

import numpy

from vector._compute.spatial import mag, theta
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
    return lib.nan_to_num(z / mag.xy_z(lib, x, y, z), nan=1.0, posinf=inf, neginf=-inf)


def xy_theta(lib, x, y, theta):
    return lib.cos(theta)


def xy_eta(lib, x, y, eta):
    return lib.cos(theta.xy_eta(lib, x, y, eta))


def rhophi_z(lib, rho, phi, z):
    return lib.nan_to_num(
        z / mag.rhophi_z(lib, rho, phi, z), nan=1.0, posinf=inf, neginf=-inf
    )


def rhophi_theta(lib, rho, phi, theta):
    return lib.cos(theta)


def rhophi_eta(lib, rho, phi, eta):
    return lib.cos(theta.rhophi_eta(lib, rho, phi, eta))


dispatch_map = {
    (AzimuthalXY, LongitudinalZ): (xy_z, float),
    (AzimuthalXY, LongitudinalTheta): (xy_theta, float),
    (AzimuthalXY, LongitudinalEta): (xy_eta, float),
    (AzimuthalRhoPhi, LongitudinalZ): (rhophi_z, float),
    (AzimuthalRhoPhi, LongitudinalTheta): (rhophi_theta, float),
    (AzimuthalRhoPhi, LongitudinalEta): (rhophi_eta, float),
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
