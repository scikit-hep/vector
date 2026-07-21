# Copyright (c) 2019, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import builtins
import typing

import numpy

from vector._methods import _check_coordinate_names


def _recname(is_momentum: bool, dimension: int) -> str:
    name = "Momentum" if is_momentum else "Vector"
    return f"{name}{dimension}D"


def _check_names(
    projectable: typing.Any, fieldnames: list[str]
) -> tuple[bool, int, list[str], typing.Any]:
    """
    Determines the record name and the columns of an array of vectors from its
    field names, allowing fields that are not coordinates to be carried along.
    """
    is_momentum, dimension, coordinates, extra = _check_coordinate_names(
        tuple(fieldnames), allow_extra=True
    )

    names = [name for name, _ in coordinates] + list(extra)
    columns = [projectable[given] for _, given in coordinates] + [
        projectable[name] for name in extra
    ]

    return is_momentum, dimension, names, columns


def _is_type_safe(array_type: typing.Any) -> None:
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
        array_type = array_type.content

    msg = "a coordinate must be of the type int or float"
    if not isinstance(array_type, awkward.types.RecordType):
        raise TypeError(msg)

    contents = array_type.contents
    for field_type in contents:
        if isinstance(field_type, awkward.types.OptionType):
            field_type = (  # noqa: PLW2901
                field_type.content
            )
        if not isinstance(field_type, awkward.types.NumpyType):
            raise TypeError(msg)
        dt = field_type.primitive
        if (
            not dt.startswith("int")
            and not dt.startswith("uint")
            and not dt.startswith("float")
        ):
            raise TypeError(msg)


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

    A coordinate may be given only once, whether by its generic name or through
    a momentum-alias, and the names must form exactly one of the combinations
    above; anything else raises a ``TypeError``. Names that are not coordinates
    become extra fields of the records.

    No constraints are placed on the types of the vector fields, though if they
    are not numbers, mathematical operations will fail. Usually, you want them to be
    integers or floating-point numbers.
    """
    import awkward

    import vector
    import vector.backends.awkward

    # convert plain Python/NumPy containers (lists, dicts of columns, ndarrays)
    # to an ak.Array, but pass through arrays that are already ak.Array or are
    # foreign array types (e.g. dask_awkward arrays), which must not be coerced.
    akarray = (
        awkward.Array(*args, **kwargs)
        if isinstance(args[0], (list, dict, numpy.ndarray))
        else args[0]
    )
    array_type = akarray.type

    _is_type_safe(array_type)

    fields = awkward.fields(akarray)

    is_momentum, dimension, names, arrays = _check_names(akarray, fields.copy())

    return awkward.with_name(
        awkward.zip(
            dict(builtins.zip(names, arrays, strict=True)),
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

    A coordinate may be given only once, whether by its generic name or through
    a momentum-alias, and the names must form exactly one of the combinations
    above; anything else raises a ``TypeError``. Names that are not coordinates
    become extra fields of the records.
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
        dict(builtins.zip(names, columns, strict=True)),
        depth_limit=depth_limit,
        with_name=_recname(is_momentum, dimension),
        behavior=behavior,
    )
