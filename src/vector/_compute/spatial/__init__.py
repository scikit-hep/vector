# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
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

import vector._compute.spatial.add  # noqa: F401
import vector._compute.spatial.costheta  # noqa: F401
import vector._compute.spatial.cottheta  # noqa: F401
import vector._compute.spatial.cross  # noqa: F401
import vector._compute.spatial.deltaangle  # noqa: F401
import vector._compute.spatial.deltaeta  # noqa: F401
import vector._compute.spatial.deltaR  # noqa: F401
import vector._compute.spatial.deltaR2  # noqa: F401
import vector._compute.spatial.dot  # noqa: F401
import vector._compute.spatial.equal  # noqa: F401
import vector._compute.spatial.eta  # noqa: F401
import vector._compute.spatial.is_antiparallel  # noqa: F401
import vector._compute.spatial.is_parallel  # noqa: F401
import vector._compute.spatial.is_perpendicular  # noqa: F401
import vector._compute.spatial.isclose  # noqa: F401
import vector._compute.spatial.mag  # noqa: F401
import vector._compute.spatial.mag2  # noqa: F401
import vector._compute.spatial.not_equal  # noqa: F401
import vector._compute.spatial.rotate_axis  # noqa: F401
import vector._compute.spatial.rotate_euler  # noqa: F401
import vector._compute.spatial.rotate_quaternion  # noqa: F401
import vector._compute.spatial.rotateX  # noqa: F401
import vector._compute.spatial.rotateY  # noqa: F401
import vector._compute.spatial.scale  # noqa: F401
import vector._compute.spatial.subtract  # noqa: F401
import vector._compute.spatial.theta  # noqa: F401
import vector._compute.spatial.transform3D  # noqa: F401
import vector._compute.spatial.unit  # noqa: F401
import vector._compute.spatial.z  # noqa: F401
