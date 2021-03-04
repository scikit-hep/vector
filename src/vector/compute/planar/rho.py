# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import vector.compute.planar.rho
import vector.geometry


def xy(lib, x1, y1):
    return lib.sqrt(vector.compute.planar.rho2.xy(lib, x1, y1))


def rhophi(lib, rho1, phi1):
    return rho1


dispatch_map = {
    (vector.geometry.AzimuthalXY,): xy,
    (vector.geometry.AzimuthalRhoPhi,): rhophi,
}


def dispatch(v1):
    return dispatch_map[
        vector.geometry.aztype(v1),
    ](v1.lib, *v1.azimuthal)
