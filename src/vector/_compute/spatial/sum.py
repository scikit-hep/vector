# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Spatial.sum(self)
"""
from __future__ import annotations

import typing

import numpy

from vector._compute.planar import x, y, rho, phi
from vector._compute.spatial import eta, theta, z
from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    _aztype,
    _flavor_of,
    _from_signature,
    _handler_of,
    _lib_of,
    _ltype,
)


# planar
def xy(lib, x_v, y_v):
    return lib.sum(x_v, axis=-1), lib.sum(y_v, axis=-1)


def xy_z(lib, x_v, y_v, z_v):
    x_u, y_u = xy(lib, x_v, y_v)
    return x_u, y_u, lib.sum(z_v, axis=-1)


def rhophi(lib, rho_v, phi_v):
    x_u, y_u = xy(lib, x.rhophi(lib, rho_v, phi_v), y.rhophi(lib, rho_v, phi_v))
    return rho.xy(lib, x_u, y_u), phi.xy(lib, x_u, y_u)


def rhophi_z(lib, rho_v, phi_v, z_v):
    rho_u, phi_u = rhophi(lib, rho_v, phi_v)
    return rho_u, phi_u, lib.sum(z_v, axis=-1)


def rhophi_theta(lib, rho_v, phi_v, theta_v):
    rho_u, phi_u, z_u = rhophi_z(
        lib, rho_v, phi_v, z.rhophi_theta(lib, rho_v, phi_v, theta_v)
    )

    return (
        rho_u,
        phi_u,
        theta.rhophi_z(lib, rho_u, phi_u, z_u),
    )


def rhophi_eta(lib, rho_v, phi_v, eta_v):
    rho_u, phi_u, z_u = rhophi_z(
        lib, rho_v, phi_v, z.rhophi_eta(lib, rho_v, phi_v, eta_v)
    )

    return (
        rho_u,
        phi_u,
        eta.rhophi_z(lib, rho_u, phi_u, z_u),
    )


def xy_theta(lib, x_v, y_v, theta_v):
    x_u, y_u, z_u = xy_z(lib, x_v, y_v, z.xy_theta(lib, x_v, y_v, theta_v))
    return x_u, y_u, theta.xy_z(lib, x_u, y_u, z_u)


def xy_eta(lib, x_v, y_v, eta_v):
    x_u, y_u, z_u = xy_z(lib, x_v, y_v, z.xy_eta(lib, x_v, y_v, eta_v))
    return (x_u, y_u, eta.xy_z(lib, x_u, y_u, z_u))


dispatch_map = {
    (AzimuthalXY, LongitudinalZ): (
        xy_z,
        AzimuthalXY,
        LongitudinalZ,
    ),
    (AzimuthalXY, LongitudinalTheta): (
        xy_theta,
        AzimuthalXY,
        LongitudinalTheta,
    ),
    (AzimuthalXY, LongitudinalEta): (
        xy_eta,
        AzimuthalXY,
        LongitudinalEta,
    ),
    (AzimuthalRhoPhi, LongitudinalZ): (
        rhophi_z,
        AzimuthalRhoPhi,
        LongitudinalZ,
    ),
    (AzimuthalRhoPhi, LongitudinalTheta): (
        rhophi_theta,
        AzimuthalRhoPhi,
        LongitudinalTheta,
    ),
    (AzimuthalRhoPhi, LongitudinalEta): (
        rhophi_eta,
        AzimuthalRhoPhi,
        LongitudinalEta,
    ),
}


def dispatch(v1: typing.Any) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (
            _aztype(v1),
            _ltype(v1),
        ),
    )
    with numpy.errstate(all="ignore"):
        return _handler_of(v1)._wrap_result(
            _flavor_of(v1),
            function(
                _lib_of(v1),
                *v1.azimuthal.elements,
                *v1.longitudinal.elements,
            ),
            returns,
            1,
        )
