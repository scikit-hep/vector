# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
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

from __future__ import annotations

import vector._compute.lorentz.add
import vector._compute.lorentz.beta
import vector._compute.lorentz.boost_beta3
import vector._compute.lorentz.boost_p4
import vector._compute.lorentz.boostX_beta
import vector._compute.lorentz.boostX_gamma
import vector._compute.lorentz.boostY_beta
import vector._compute.lorentz.boostY_gamma
import vector._compute.lorentz.boostZ_beta
import vector._compute.lorentz.boostZ_gamma
import vector._compute.lorentz.deltaRapidityPhi
import vector._compute.lorentz.deltaRapidityPhi2
import vector._compute.lorentz.dot
import vector._compute.lorentz.equal
import vector._compute.lorentz.Et
import vector._compute.lorentz.Et2
import vector._compute.lorentz.gamma
import vector._compute.lorentz.is_lightlike
import vector._compute.lorentz.is_spacelike
import vector._compute.lorentz.is_timelike
import vector._compute.lorentz.isclose
import vector._compute.lorentz.Mt
import vector._compute.lorentz.Mt2
import vector._compute.lorentz.not_equal
import vector._compute.lorentz.rapidity
import vector._compute.lorentz.scale
import vector._compute.lorentz.subtract
import vector._compute.lorentz.t
import vector._compute.lorentz.t2
import vector._compute.lorentz.tau
import vector._compute.lorentz.tau2
import vector._compute.lorentz.to_beta3
import vector._compute.lorentz.transform4D
import vector._compute.lorentz.unit  # noqa: F401
