# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import vector.geometry


def xy(lib, x1, y1):
    return y1


def rhophi(lib, rho1, phi1):
    return rho1 * lib.sin(phi1)


dispatch_map = {
    (vector.geometry.AzimuthalXY,): xy,
    (vector.geometry.AzimuthalRhoPhi,): rhophi,
}


def dispatch(lib, v1):
    return dispatch_map[
        vector.geometry.aztype(v1),
    ](lib, *v1.azimuthal)
