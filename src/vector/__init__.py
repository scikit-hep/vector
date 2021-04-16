# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import sys
import typing

import packaging.version

from vector._backends.awkward_constructors import Array as Array
from vector._backends.awkward_constructors import Array as arr
from vector._backends.awkward_constructors import Array as awk
from vector._backends.awkward_constructors import zip
from vector._backends.numpy_ import (
    MomentumNumpy2D,
    MomentumNumpy3D,
    MomentumNumpy4D,
    VectorNumpy,
    VectorNumpy2D,
    VectorNumpy3D,
    VectorNumpy4D,
    array,
)
from vector._backends.object_ import (
    MomentumObject2D,
    MomentumObject3D,
    MomentumObject4D,
    VectorObject,
    VectorObject2D,
    VectorObject3D,
    VectorObject4D,
    obj,
)
from vector._methods import (
    Azimuthal,
    AzimuthalRhoPhi,
    AzimuthalXY,
    Coordinates,
    Longitudinal,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    Lorentz,
    Momentum,
    Planar,
    Spatial,
    Temporal,
    TemporalT,
    TemporalTau,
    Vector,
    Vector2D,
    Vector3D,
    Vector4D,
    dim,
)
from vector.version import version as __version__

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


def _import_awkward() -> None:
    awk_version = packaging.version.Version(importlib_metadata.version("awkward"))
    if awk_version < packaging.version.Version("1.2.0rc5"):
        # the only context users will see this message is if they're trying to use vector.awk
        # VectorAwkward is still set to None
        msg = f"awkward {awk_version} is too old; please upgrade to 1.2.0 or later"
        raise ImportError(msg)


try:
    import awkward

    _import_awkward()
except ImportError:
    awkward = None
    if not typing.TYPE_CHECKING:
        VectorAwkward = None
else:
    from vector._backends.awkward_ import VectorAwkward


__all__: typing.Tuple[str, ...] = (
    "Array",
    "Azimuthal",
    "AzimuthalRhoPhi",
    "AzimuthalXY",
    "Coordinates",
    "Longitudinal",
    "LongitudinalEta",
    "LongitudinalTheta",
    "LongitudinalZ",
    "Lorentz",
    "Momentum",
    "MomentumNumpy2D",
    "MomentumNumpy3D",
    "MomentumNumpy4D",
    "MomentumObject2D",
    "MomentumObject3D",
    "MomentumObject4D",
    "Planar",
    "Spatial",
    "Temporal",
    "TemporalT",
    "TemporalTau",
    "Vector",
    "Vector2D",
    "Vector3D",
    "Vector4D",
    "VectorAwkward",
    "VectorNumpy",
    "VectorNumpy2D",
    "VectorNumpy3D",
    "VectorNumpy4D",
    "VectorObject",
    "VectorObject2D",
    "VectorObject3D",
    "VectorObject4D",
    "__version__",
    "arr",
    "array",
    "awk",
    "zip",
    "dim",
    "obj",
    "register_awkward",
    "register_numba",
)


def __dir__() -> typing.Tuple[str, ...]:
    return (
        tuple(s for s in __all__ if s != "VectorAwkward")
        if awkward is None
        else __all__
    )


def register_numba() -> None:
    """
    Make Vector types known to Numba's compiler, so that JIT-compilations with
    vector arguments do not fail.

    This usually isn't necessary, as it is passed to Numba's ``entry_point`` and
    is therefore executed as soon as Numba is imported.
    """
    import vector._backends.numba_numpy  # noqa: 401
    import vector._backends.numba_object  # noqa: 401


_awkward_registered = False


def register_awkward() -> None:
    """
    Make Vector behaviors known to Awkward Array's ``ak.behavior`` mechanism.

    If you call this function, any records named ``Vector2D``, ``Vector3D``,
    ``Vector4D``, ``Momentum2D``, ``Momentum3D``, and ``Momentum4D`` will have
    vector properties and methods.

    If you do not call this function, only arrays created with
    :doc:`vector.Array` (or derived from such an array) have vector properties
    and methods.
    """
    import awkward

    import vector._backends.awkward_  # noqa: 401

    global _awkward_registered
    awkward.behavior.update(vector._backends.awkward_.behavior)
    _awkward_registered = True
