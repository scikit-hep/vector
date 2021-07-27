import typing

"""
.. code-block:: python

    Spatial.contravariant(self, other)
"""

import numpy

from vector._compute.planar import add, x, y
from vector._compute.spatial import eta, theta, z
from vector._methods import (
    AzimuthalRhoPhi,
    AzimuthalXY,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    _aztype,
    _flavor_of,
    _from_signature,
    _handler_of,
    _lib_of,
    _ltype,
)