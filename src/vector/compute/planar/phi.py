# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import vector.geometry


def xy(lib, x1, y1):
    return lib.arctan2(y1, x1)


def rhophi(lib, rho1, phi1):
    return phi1


dispatch_map = {
    (vector.geometry.AzimuthalXY,): xy,
    (vector.geometry.AzimuthalRhoPhi,): rhophi,
}


def dispatch(lib, v1):
    for azimuthal_type in type(v1.azimuthal).__mro__:
        if azimuthal_type.__module__ == "vector.geometry":
            break
    return dispatch_map[
        azimuthal_type,
    ](lib, *v1.azimuthal)
