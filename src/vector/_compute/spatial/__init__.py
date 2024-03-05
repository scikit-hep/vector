# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
Compute functions for spatial vectors, which is to say 3D and 4D.

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

from __future__ import annotations

import vector._compute.spatial.add
import vector._compute.spatial.costheta
import vector._compute.spatial.cottheta
import vector._compute.spatial.cross
import vector._compute.spatial.deltaangle
import vector._compute.spatial.deltaeta
import vector._compute.spatial.deltaR
import vector._compute.spatial.deltaR2
import vector._compute.spatial.dot
import vector._compute.spatial.equal
import vector._compute.spatial.eta
import vector._compute.spatial.is_antiparallel
import vector._compute.spatial.is_parallel
import vector._compute.spatial.is_perpendicular
import vector._compute.spatial.isclose
import vector._compute.spatial.mag
import vector._compute.spatial.mag2
import vector._compute.spatial.not_equal
import vector._compute.spatial.rotate_axis
import vector._compute.spatial.rotate_euler
import vector._compute.spatial.rotate_quaternion
import vector._compute.spatial.rotateX
import vector._compute.spatial.rotateY
import vector._compute.spatial.scale
import vector._compute.spatial.subtract
import vector._compute.spatial.theta
import vector._compute.spatial.transform3D
import vector._compute.spatial.unit
import vector._compute.spatial.z  # noqa: F401
