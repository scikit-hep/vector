# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
.. code-block:: python

    @property
    Spatial.eta(self)
"""

from __future__ import annotations

import typing
from math import inf

import numpy

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


# TODO: https://github.com/scikit-hep/vector/issues/615
# revert back to `nan_to_num` implementation once
# https://github.com/cupy/cupy/issues/9143 is fixed.
# `lib.where` works but there is no SymPy equivalent for the function.
def xy_z(lib, x, y, z):
    return (
        lib.where(
            z != 0, lib.arcsinh(lib.where(z != 0, z / lib.sqrt(x**2 + y**2), z)), z
        )
        * 1
    )


def xy_theta(lib, x, y, theta):
    return lib.nan_to_num(
        -lib.log(lib.tan(0.5 * theta)), nan=0.0, posinf=inf, neginf=-inf
    )


def xy_eta(lib, x, y, eta):
    return eta


xy_eta.__awkward_transform_allowed__ = False  # type:ignore[attr-defined]


# TODO: https://github.com/scikit-hep/vector/issues/615
# revert back to `nan_to_num` implementation once
# https://github.com/cupy/cupy/issues/9143 is fixed.
# `lib.where` works but there is no SymPy equivalent for the function.
def rhophi_z(lib, rho, phi, z):
    return lib.where(z != 0, lib.arcsinh(lib.where(z != 0, z / rho, z)), z) * 1


def rhophi_theta(lib, rho, phi, theta):
    return -lib.log(lib.tan(0.5 * theta))


def rhophi_eta(lib, rho, phi, eta):
    return eta


rhophi_eta.__awkward_transform_allowed__ = False  # type:ignore[attr-defined]


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
            v._wrap_dispatched_function(function)(
                v.lib, *v.azimuthal.elements, *v.longitudinal.elements
            ),
            returns,
            1,
        )
