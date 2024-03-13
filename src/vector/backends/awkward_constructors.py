# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import typing

import numpy


def _recname(is_momentum: bool, dimension: int) -> str:
    name = "Momentum" if is_momentum else "Vector"
    return f"{name}{dimension}D"


def _check_names(
    projectable: typing.Any, fieldnames: list[str]
) -> tuple[bool, int, list[str], typing.Any]:
    complaint1 = "duplicate coordinates (through momentum-aliases): " + ", ".join(
        repr(x) for x in fieldnames
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
    columns = []

    if "x" in fieldnames and "y" in fieldnames:
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["x", "y"])
        columns.extend([projectable["x"], projectable["y"]])
        fieldnames.remove("x")
        fieldnames.remove("y")
    if "rho" in fieldnames and "phi" in fieldnames:
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["rho", "phi"])
        columns.extend([projectable["rho"], projectable["phi"]])
        fieldnames.remove("rho")
        fieldnames.remove("phi")
    if "x" in fieldnames and "py" in fieldnames:
        is_momentum = True
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["x", "y"])
        columns.extend([projectable["x"], projectable["py"]])
        fieldnames.remove("x")
        fieldnames.remove("py")
    if "px" in fieldnames and "y" in fieldnames:
        is_momentum = True
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["x", "y"])
        columns.extend([projectable["px"], projectable["y"]])
        fieldnames.remove("px")
        fieldnames.remove("y")
    if "px" in fieldnames and "py" in fieldnames:
        is_momentum = True
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["x", "y"])
        columns.extend([projectable["px"], projectable["py"]])
        fieldnames.remove("px")
        fieldnames.remove("py")
    if "pt" in fieldnames and "phi" in fieldnames:
        is_momentum = True
        if dimension != 0:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 2
        names.extend(["rho", "phi"])
        columns.extend([projectable["pt"], projectable["phi"]])
        fieldnames.remove("pt")
        fieldnames.remove("phi")

    if "z" in fieldnames:
        if dimension != 2:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 3
        names.append("z")
        columns.append(projectable["z"])
        fieldnames.remove("z")
    if "theta" in fieldnames:
        if dimension != 2:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 3
        names.append("theta")
        columns.append(projectable["theta"])
        fieldnames.remove("theta")
    if "eta" in fieldnames:
        if dimension != 2:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 3
        names.append("eta")
        columns.append(projectable["eta"])
        fieldnames.remove("eta")
    if "pz" in fieldnames:
        is_momentum = True
        if dimension != 2:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 3
        names.append("z")
        columns.append(projectable["pz"])
        fieldnames.remove("pz")

    if "t" in fieldnames:
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("t")
        columns.append(projectable["t"])
        fieldnames.remove("t")
    if "tau" in fieldnames:
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("tau")
        columns.append(projectable["tau"])
        fieldnames.remove("tau")
    if "E" in fieldnames:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("t")
        columns.append(projectable["E"])
        fieldnames.remove("E")
    if "e" in fieldnames:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("t")
        columns.append(projectable["e"])
        fieldnames.remove("e")
    if "energy" in fieldnames:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("t")
        columns.append(projectable["energy"])
        fieldnames.remove("energy")
    if "M" in fieldnames:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("tau")
        columns.append(projectable["M"])
        fieldnames.remove("M")
    if "m" in fieldnames:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("tau")
        columns.append(projectable["m"])
        fieldnames.remove("m")
    if "mass" in fieldnames:
        is_momentum = True
        if dimension != 3:
            raise TypeError(complaint1 if is_momentum else complaint2)
        dimension = 4
        names.append("tau")
        columns.append(projectable["mass"])
        fieldnames.remove("mass")

    if dimension == 0:
        raise TypeError(complaint1 if is_momentum else complaint2)

    for name in fieldnames:
        names.append(name)
        columns.append(projectable[name])

    return is_momentum, dimension, names, columns


def _is_type_safe(array_type: typing.Any) -> bool:
    import awkward

    while isinstance(
        array_type,
        (
            awkward.types.ArrayType,
            awkward.types.RegularType,
            awkward.types.ListType,
            awkward.types.OptionType,
        ),
    ):
        # .content is Awkward v2
        array_type = (
            array_type.content if hasattr(array_type, "content") else array_type.type
        )

    if not isinstance(array_type, awkward.types.RecordType):
        return False

    # .contents is Awkward v2
    contents = (
        array_type.contents if hasattr(array_type, "contents") else array_type.fields()
    )
    for field_type in contents:
        if isinstance(field_type, awkward.types.OptionType):
            field_type = (  # noqa: PLW2901
                field_type.content
                if hasattr(array_type, "content")
                else field_type.type
            )
        if not isinstance(
            field_type,
            awkward.types.NumpyType
            if hasattr(awkward.types, "NumpyType")
            else awkward.types.PrimitiveType,
        ):
            return False
        dt = (
            field_type.primitive
            if hasattr(field_type, "primitive")
            else field_type.dtype
        )
        if (
            not dt.startswith("int")
            and not dt.startswith("uint")
            and not dt.startswith("float")
        ):
            return False

    return True


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
    - ``e`` may be substituted for ``t``
    - ``energy`` may be substituted for ``t``
    - ``M`` may be substituted for ``tau``
    - ``m`` may be substituted for ``tau``
    - ``mass`` may be substituted for ``tau``

    to make the vector a momentum vector.

    No constraints are placed on the types of the vector fields, though if they
    are not numbers, mathematical operations will fail. Usually, you want them to be
    integers or floating-point numbers.
    """
    import awkward

    import vector
    import vector.backends.awkward

    # don't pass dask_awkward arrays in ak.Array
    akarray = (
        awkward.Array(*args, **kwargs)
        if isinstance(args[0], (list, numpy.ndarray))
        else args[0]
    )
    array_type = akarray.type

    if not _is_type_safe(array_type):
        raise TypeError("a coordinate must be of the type int or float")
    fields = awkward.fields(akarray)

    is_momentum, dimension, names, arrays = _check_names(akarray, fields.copy())

    # don't execute for dask_awkward arrays
    if isinstance(args[0], (awkward.Array, list, numpy.ndarray)):
        needs_behavior = not vector._awkward_registered
        for x in arrays:
            if needs_behavior:
                if x.behavior is None:
                    x.behavior = vector.backends.awkward.behavior
                else:
                    x.behavior = dict(x.behavior)
                    x.behavior.update(vector.backends.awkward.behavior)
            else:
                x.behavior = None
            needs_behavior = False

    assert 2 <= dimension <= 4, f"Dimension must be between 2-4, not {dimension}"

    return awkward.with_name(
        awkward.zip(
            dict(__builtins__["zip"](names, arrays)),  # type: ignore[index]
            depth_limit=akarray.layout.purelist_depth,
        ),
        _recname(is_momentum, dimension),
        behavior=vector.backends.awkward.behavior,
    )


def zip(arrays: dict[str, typing.Any], depth_limit: int | None = None) -> typing.Any:
    """
    Constructs an Awkward Array of vectors, whose type is determined by the fields
    of the record array (which may be nested within lists or other non-record structures).

    This function accepts a subset of ``ak.zip``'s arguments.

    Args:

        arrays (dict of str to array-like): Arrays, lists, etc. to zip together.
            Unlike ``ak.zip``, this must be a dict with string keys to determine
            the coordinate system of the arrays; it may not be a tuple.
        depth_limit (None or int): If None, attempt to fully broadcast the
            ``array`` to all levels. If an int, limit the number of dimensions
            that get broadcasted. The minimum value is ``1``, for no broadcasting.

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
    - ``e`` may be substituted for ``t``
    - ``energy`` may be substituted for ``t``
    - ``M`` may be substituted for ``tau``
    - ``m`` may be substituted for ``tau``
    - ``mass`` may be substituted for ``tau``

    to make the vector a momentum vector.
    """
    import awkward

    import vector
    import vector.backends.awkward

    if not isinstance(arrays, dict):
        raise TypeError("argument passed to vector.zip must be a dictionary")

    is_momentum, dimension, names, columns = _check_names(arrays, list(arrays.keys()))

    behavior = None
    if not vector._awkward_registered:
        behavior = dict(vector.backends.awkward.behavior)

    return awkward.zip(
        dict(__builtins__["zip"](names, columns)),  # type: ignore[index]
        depth_limit=depth_limit,
        with_name=_recname(is_momentum, dimension),
        behavior=behavior,
    )
