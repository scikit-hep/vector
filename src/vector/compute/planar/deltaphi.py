# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

from vector.geometry import AzimuthalRhoPhi, AzimuthalXY, aztype
from vector.compute.planar import phi


def xy_xy(lib, x1, y1, x2, y2):
    return phi.xy(lib, x1, y1) - phi.xy(lib, x2, y2)


def xy_rhophi(lib, x1, y1, rho2, phi2):
    return phi.xy(lib, x1, y1) - phi2


def rhophi_xy(lib, rho1, phi1, x2, y2):
    return phi1 - phi.xy(lib, x2, y2)


def rhophi_rhophi(lib, rho1, phi1, rho2, phi2):
    return phi1 - phi2


dispatch_map = {
    (AzimuthalXY, AzimuthalXY): xy_xy,
    (AzimuthalXY, AzimuthalRhoPhi): xy_rhophi,
    (AzimuthalRhoPhi, AzimuthalXY): rhophi_xy,
    (AzimuthalRhoPhi, AzimuthalRhoPhi): rhophi_rhophi,
}


def dispatch(v1, v2):
    with numpy.errstate(all="ignore"):
        return dispatch_map[
            aztype(v1),
            aztype(v2),
        ](v1.lib, *v1.azimuthal.elements, *v2.azimuthal.elements)
