# Copyright (c) 2019-2023, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    Planar.sum(self)
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


def rhophi(lib, rho_v, phi_v):
    x_u, y_u = xy(lib, x.rhophi(lib, rho_v, phi_v), y.rhophi(lib, rho_v, phi_v))
    return rho.xy(lib, x_u, y_u), phi.xy(lib, x_u, y_u)


dispatch_map = {
    (AzimuthalXY,): (
        xy,
        AzimuthalXY,
    ),
    (AzimuthalRhoPhi,): (
        rhophi,
        AzimuthalRhoPhi,
    ),
}


def dispatch(v1: typing.Any) -> typing.Any:
    function, *returns = _from_signature(
        __name__,
        dispatch_map,
        (_aztype(v1),),
    )
    with numpy.errstate(all="ignore"):
        return _handler_of(v1)._wrap_result(
            _flavor_of(v1),
            function(
                _lib_of(v1),
                *v1.azimuthal.elements,
            ),
            returns,
            1,
        )
