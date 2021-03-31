# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
Compute functions for lorentz vectors, which is to say 4D.

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

import vector._compute.lorentz.add  # noqa: F401
import vector._compute.lorentz.beta  # noqa: F401
import vector._compute.lorentz.boost_beta3  # noqa: F401
import vector._compute.lorentz.boost_p4  # noqa: F401
import vector._compute.lorentz.boostX_beta  # noqa: F401
import vector._compute.lorentz.boostX_gamma  # noqa: F401
import vector._compute.lorentz.boostY_beta  # noqa: F401
import vector._compute.lorentz.boostY_gamma  # noqa: F401
import vector._compute.lorentz.boostZ_beta  # noqa: F401
import vector._compute.lorentz.boostZ_gamma  # noqa: F401
import vector._compute.lorentz.dot  # noqa: F401
import vector._compute.lorentz.equal  # noqa: F401
import vector._compute.lorentz.Et  # noqa: F401
import vector._compute.lorentz.Et2  # noqa: F401
import vector._compute.lorentz.gamma  # noqa: F401
import vector._compute.lorentz.is_lightlike  # noqa: F401
import vector._compute.lorentz.is_spacelike  # noqa: F401
import vector._compute.lorentz.is_timelike  # noqa: F401
import vector._compute.lorentz.isclose  # noqa: F401
import vector._compute.lorentz.Mt  # noqa: F401
import vector._compute.lorentz.Mt2  # noqa: F401
import vector._compute.lorentz.not_equal  # noqa: F401
import vector._compute.lorentz.rapidity  # noqa: F401
import vector._compute.lorentz.scale  # noqa: F401
import vector._compute.lorentz.subtract  # noqa: F401
import vector._compute.lorentz.t  # noqa: F401
import vector._compute.lorentz.t2  # noqa: F401
import vector._compute.lorentz.tau  # noqa: F401
import vector._compute.lorentz.tau2  # noqa: F401
import vector._compute.lorentz.to_beta3  # noqa: F401
import vector._compute.lorentz.transform4D  # noqa: F401
import vector._compute.lorentz.unit  # noqa: F401
