# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import typing

from vector._backends.awkward_ import VectorAwkward
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

__all__ = (
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
    "dim",
    "obj",
    "register_awkward",
    "register_numba",
)


def __dir__() -> typing.Tuple[str, ...]:
    return __all__


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


def Array(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """
    Constructs an Awkward Array of vectors, whose type is determined by the fields
    of the record array (which may be nested within lists or other non-record structures).

    All allowed signatures for ``ak.Array`` can be used in this function.

    The array must contain records with the following combinations of field names:

    - (2D) ``x``, ``y``
    - (2D) ``rho``, ``phi``
    - (3D) ``x``, ``y``, ``z``
    - (3D) ``x``, ``y``, ``theta``
    - (3D) ``x``, ``y``, ``eta``
    - (3D) ``rho``, ``phi``, ``z``
    - (3D) ``rho``, ``phi``, ``theta``
    - (3D) ``rho``, ``phi``, ``eta``
    - (4D) ``x``, ``y``, ``z``, ``t``
    - (4D) ``x``, ``y``, ``z``, ``tau```
    - (4D) ``x``, ``y``, ``theta``, ``t```
    - (4D) ``x``, ``y``, ``theta``, ``tau```
    - (4D) ``x``, ``y``, ``eta``, ``t```
    - (4D) ``x``, ``y``, ``eta``, ``tau```
    - (4D) ``rho``, ``phi``, ``z``, ``t```
    - (4D) ``rho``, ``phi``, ``z``, ``tau```
    - (4D) ``rho``, ``phi``, ``theta``, ``t```
    - (4D) ``rho``, ``phi``, ``theta``, ``tau```
    - (4D) ``rho``, ``phi``, ``eta``, ``t```
    - (4D) ``rho``, ``phi``, ``eta``, ``tau```

    in which

    - ``px`` may be substituted for ``x``
    - ``py`` may be substituted for ``y``
    - ``pt`` may be substituted for ``rho``
    - ``pz`` may be substituted for ``z``
    - ``E`` may be substituted for ``t``
    - ``energy`` may be substituted for ``t``
    - ``M`` may be substituted for ``tau``
    - ``mass`` may be substituted for ``tau``

    to make the vector a momentum vector.

    No constraints are placed on the types of the vector fields, though if they
    are not numbers, mathematical operations will fail. Usually, you want them to be
    integers or floating-point numbers.
    """
    import awkward

    import vector._backends.awkward_  # noqa: 401

    akarray = awkward.Array(*args, **kwargs)
    fields = awkward.fields(akarray)

    complaint1 = "duplicate coordinates (through momentum-aliases): " + ", ".join(
        repr(x) for x in fields
    )
    complaint2 = (
        "unrecognized combination of coordinates, allowed combinations are:\n\n"
        "    (2D) x= y=\n"
        "    (2D) rho= phi=\n"
        "    (3D) x= y= z=\n"
        "    (3D) x= y= theta=\n"
        "    (3D) x= y= eta=\n"
        "    (3D) rho= phi= z=\n"
        "    (3D) rho= phi= theta=\n"
        "    (3D) rho= phi= eta=\n"
        "    (4D) x= y= z= t=\n"
        "    (4D) x= y= z= tau=\n"
        "    (4D) x= y= theta= t=\n"
        "    (4D) x= y= theta= tau=\n"
        "    (4D) x= y= eta= t=\n"
        "    (4D) x= y= eta= tau=\n"
        "    (4D) rho= phi= z= t=\n"
        "    (4D) rho= phi= z= tau=\n"
        "    (4D) rho= phi= theta= t=\n"
        "    (4D) rho= phi= theta= tau=\n"
        "    (4D) rho= phi= eta= t=\n"
        "    (4D) rho= phi= eta= tau="
    )

    is_momentum = False
    dimension = 0
    names = []
    arrays = []

    if "x" in fields and "y" in fields:
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["x", "y"])
        arrays.extend([akarray["x"], akarray["y"]])
        fields.remove("x")
        fields.remove("y")
    if "rho" in fields and "phi" in fields:
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["rho", "phi"])
        arrays.extend([akarray["rho"], akarray["phi"]])
        fields.remove("rho")
        fields.remove("phi")
    if "x" in fields and "py" in fields:
        is_momentum = True
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["x", "y"])
        arrays.extend([akarray["x"], akarray["py"]])
        fields.remove("x")
        fields.remove("py")
    if "px" in fields and "y" in fields:
        is_momentum = True
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["x", "y"])
        arrays.extend([akarray["px"], akarray["y"]])
        fields.remove("px")
        fields.remove("y")
    if "px" in fields and "py" in fields:
        is_momentum = True
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["x", "y"])
        arrays.extend([akarray["px"], akarray["py"]])
        fields.remove("px")
        fields.remove("py")
    if "pt" in fields and "phi" in fields:
        is_momentum = True
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["rho", "phi"])
        arrays.extend([akarray["pt"], akarray["phi"]])
        fields.remove("pt")
        fields.remove("phi")

    if "z" in fields:
        if dimension != 2:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 3
        names.append("z")
        arrays.append(akarray["z"])
        fields.remove("z")
    if "theta" in fields:
        if dimension != 2:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 3
        names.append("theta")
        arrays.append(akarray["theta"])
        fields.remove("theta")
    if "eta" in fields:
        if dimension != 2:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 3
        names.append("eta")
        arrays.append(akarray["eta"])
        fields.remove("eta")
    if "pz" in fields:
        is_momentum = True
        if dimension != 2:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 3
        names.append("z")
        arrays.append(akarray["pz"])
        fields.remove("pz")

    if "t" in fields:
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("t")
        arrays.append(akarray["t"])
        fields.remove("t")
    if "tau" in fields:
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("tau")
        arrays.append(akarray["tau"])
        fields.remove("tau")
    if "E" in fields:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("t")
        arrays.append(akarray["E"])
        fields.remove("E")
    if "energy" in fields:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("t")
        arrays.append(akarray["energy"])
        fields.remove("energy")
    if "M" in fields:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("tau")
        arrays.append(akarray["M"])
        fields.remove("M")
    if "mass" in fields:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("tau")
        arrays.append(akarray["mass"])
        fields.remove("mass")

    if dimension == 0:
        raise TypeError(complaint1 if is_momentum else complaint2)

    for name in fields:
        names.append(name)
        arrays.append(akarray[name])

    needs_behavior = not _awkward_registered
    for x in arrays:
        if needs_behavior:
            if x.behavior is None:
                x.behavior = vector._backends.awkward_.behavior
            else:
                x.behavior = dict(x.behavior)
                x.behavior.update(vector._backends.awkward_.behavior)
        else:
            x.behavior = None
        needs_behavior = False

    depth = akarray.layout.purelist_depth

    assert 2 <= dimension <= 4, f"Dimension must be between 2-4, not {dimension}"

    name = "Momentum" if is_momentum else "Vector"
    recname = f"{name}{dimension}D"

    return awkward.zip(dict(zip(names, arrays)), depth_limit=depth, with_name=recname)


arr = Array
awk = Array
