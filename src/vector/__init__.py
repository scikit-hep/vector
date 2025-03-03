# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import importlib.metadata
import typing

import packaging.version

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
from vector._version import version as __version__
from vector.backends.awkward_constructors import Array, zip
from vector.backends.awkward_constructors import Array as awk
from vector.backends.numpy import (
    MomentumNumpy2D,
    MomentumNumpy3D,
    MomentumNumpy4D,
    VectorNumpy,
    VectorNumpy2D,
    VectorNumpy3D,
    VectorNumpy4D,
    array,
)
from vector.backends.numpy import array as arr
from vector.backends.object import (
    MomentumObject2D,
    MomentumObject3D,
    MomentumObject4D,
    VectorObject,
    VectorObject2D,
    VectorObject3D,
    VectorObject4D,
    obj,
)


def _import_awkward() -> None:
    awk_version = packaging.version.Version(importlib.metadata.version("awkward"))
    if awk_version < packaging.version.Version("2.0.0"):
        # the only context users will see this message is if they're trying to use vector.awk
        # VectorAwkward is still set to None
        msg = f"awkward {awk_version} is too old; please upgrade to 2.0.0 or later"
        raise ImportError(msg)


try:
    import awkward

    _import_awkward()
except ImportError:
    awkward = None
    if not typing.TYPE_CHECKING:
        VectorAwkward = None
else:
    from vector.backends.awkward import VectorAwkward, awkward_transform

try:
    import sympy  # type: ignore[import-untyped]
except ImportError:
    sympy = None
    if not typing.TYPE_CHECKING:
        VectorSympy = None
else:
    from vector.backends.sympy import (
        MomentumSympy2D,
        MomentumSympy3D,
        MomentumSympy4D,
        VectorSympy,
        VectorSympy2D,
        VectorSympy3D,
        VectorSympy4D,
    )


__all__: tuple[str, ...] = (
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
    "MomentumSympy2D",
    "MomentumSympy3D",
    "MomentumSympy4D",
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
    "VectorSympy",
    "VectorSympy2D",
    "VectorSympy3D",
    "VectorSympy4D",
    "__version__",
    "arr",
    "array",
    "awk",
    "awkward_transform",
    "dim",
    "obj",
    "register_awkward",
    "register_numba",
    "zip",
)


def __dir__() -> tuple[str, ...]:
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
    import vector.backends._numba_object
    import vector.backends.numba_numpy  # noqa: F401


_awkward_registered = False


def register_awkward() -> None:
    """
    Make Vector behaviors known to Awkward Array's ``ak.behavior`` mechanism.

    If you call this function, any records named ``Vector2D``, ``Vector3D``,
    ``Vector4D``, ``Momentum2D``, ``Momentum3D``, and ``Momentum4D`` will have
    vector properties and methods.

    If you do not call this function, only arrays created with
    :func:`vector.Array` (or derived from such an array) have vector properties
    and methods.
    """
    import awkward

    import vector.backends.awkward

    global _awkward_registered  # noqa: PLW0603
    awkward.behavior.update(vector.backends.awkward.behavior)
    _awkward_registered = True
