# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.compute.lorentz import dot
from vector.methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    TemporalT,
    TemporalTau,
    _from_signature, _aztype,
    _ltype,
    _ttype,
)

dispatch_map = {}


def make_function(azimuthal, longitudinal, temporal):
    dot_function, _ = dot.dispatch_map[
        azimuthal, longitudinal, temporal, azimuthal, longitudinal, temporal
    ]

    def f(lib, tolerance, coord1, coord2, coord3, coord4):
        return lib.absolute(
            dot_function(
                lib, coord1, coord2, coord3, coord4, coord1, coord2, coord3, coord4
            )
        ) < lib.absolute(tolerance)

    dispatch_map[azimuthal, longitudinal, temporal] = (f, bool)


for azimuthal in (AzimuthalXY, AzimuthalRhoPhi):
    for longitudinal in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
        for temporal in (TemporalT, TemporalTau):
            make_function(azimuthal, longitudinal, temporal)


def dispatch(tolerance, v):
    function, *returns = _from_signature(__name__, dispatch_map, (
        _aztype(v),
        _ltype(v),
        _ttype(v),
    ))
    with numpy.errstate(all="ignore"):
        return v._wrap_result(
            function(
                v.lib,
                tolerance,
                *v.azimuthal.elements,
                *v.longitudinal.elements,
                *v.temporal.elements,
            ),
            returns,
        )
