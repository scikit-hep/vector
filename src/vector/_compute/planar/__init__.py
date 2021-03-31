# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
Compute functions for planar vectors, which is to say 2D, 3D, and 4D.

Each function is a module with variants for each coordinate system (or combination
of coordinate systems) as functions within the module.

Each module has a ``dispatch_map`` (dict) that maps coordinate types to the
appropriate function and its return type(s), and a ``dispatch`` (function) uses
this information to call the right function and return the right type.

The compute functions themselves are restricted to a minimum of Python features:
no statements other than assignments and one return, no assumptions about data
types. In particular, if statements and loops are not allowed. The
tests/test_compute_features.py suite ensures that these rules are followed (though
that set of allowed features can be expanded if it doesn't prevent the addition
of new backends).
"""

import vector._compute.planar.add  # noqa: F401
import vector._compute.planar.deltaphi  # noqa: F401
import vector._compute.planar.dot  # noqa: F401
import vector._compute.planar.equal  # noqa: F401
import vector._compute.planar.is_antiparallel  # noqa: F401
import vector._compute.planar.is_parallel  # noqa: F401
import vector._compute.planar.is_perpendicular  # noqa: F401
import vector._compute.planar.isclose  # noqa: F401
import vector._compute.planar.not_equal  # noqa: F401
import vector._compute.planar.phi  # noqa: F401
import vector._compute.planar.rho  # noqa: F401
import vector._compute.planar.rho2  # noqa: F401
import vector._compute.planar.rotateZ  # noqa: F401
import vector._compute.planar.scale  # noqa: F401
import vector._compute.planar.subtract  # noqa: F401
import vector._compute.planar.transform2D  # noqa: F401
import vector._compute.planar.unit  # noqa: F401
import vector._compute.planar.x  # noqa: F401
import vector._compute.planar.y  # noqa: F401
