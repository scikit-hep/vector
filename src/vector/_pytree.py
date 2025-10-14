from __future__ import annotations

import functools
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy

if TYPE_CHECKING:
    from optree.pytree import ReexportedPyTreeModule

from vector._methods import (
    Vector2D,
    Vector3D,
    Vector4D,
)
from vector.backends.numpy import (
    MomentumNumpy2D,
    MomentumNumpy3D,
    MomentumNumpy4D,
    VectorNumpy,
    VectorNumpy2D,
    VectorNumpy3D,
    VectorNumpy4D,
)
from vector.backends.object import (
    MomentumObject2D,
    MomentumObject3D,
    MomentumObject4D,
    VectorObject2D,
    VectorObject3D,
    VectorObject4D,
)

Children = tuple[Any, ...]
MetaData = tuple[type, ...]


def _flatten2D(v: Vector2D) -> tuple[Children, MetaData]:
    children = v.azimuthal.elements
    metadata = type(v), type(v.azimuthal)
    return children, metadata


def _unflatten2D(metadata: MetaData, children: Children) -> Vector2D:
    backend, azimuthal = metadata
    return backend(azimuthal=azimuthal(*children))


def _flatten3D(v: Vector3D) -> tuple[Children, MetaData]:
    children = v.azimuthal.elements, v.longitudinal.elements
    metadata = type(v), type(v.azimuthal), type(v.longitudinal)
    return children, metadata


def _unflatten3D(metadata: MetaData, children: Children) -> Vector3D:
    coords_azimuthal, coords_longitudinal = children
    backend, azimuthal, longitudinal = metadata
    return backend(
        azimuthal=azimuthal(*coords_azimuthal),
        longitudinal=longitudinal(*coords_longitudinal),
    )


def _flatten4D(v: Vector4D) -> tuple[Children, MetaData]:
    children = (
        v.azimuthal.elements,
        v.longitudinal.elements,
        v.temporal.elements,
    )
    metadata = type(v), type(v.azimuthal), type(v.longitudinal), type(v.temporal)
    return children, metadata


def _unflatten4D(metadata: MetaData, children: Children) -> Vector4D:
    coords_azimuthal, coords_longitudinal, coords_temporal = children
    backend, azimuthal, longitudinal, temporal = metadata
    return backend(
        azimuthal=azimuthal(*coords_azimuthal),
        longitudinal=longitudinal(*coords_longitudinal),
        temporal=temporal(*coords_temporal),
    )


def _flattenAoSdata(v: VectorNumpy) -> tuple[Children, tuple[type, numpy.dtype]]:
    assert v.dtype.fields is not None
    field_dtypes = [dt for dt, *_ in v.dtype.fields.values()]
    target_dtype = field_dtypes[0]
    if not all(fd == target_dtype for fd in field_dtypes):
        raise ValueError("All fields must have the same dtype to flatten")
    array = numpy.array(v).view(target_dtype)
    children = (array,)
    metadata = (type(v), v.dtype)
    return children, metadata


def _unflattenAoSdata(
    metadata: tuple[type, numpy.dtype], children: Children
) -> VectorNumpy:
    (array,) = children
    (vtype, dtype) = metadata
    return array.view(dtype).view(vtype)


@functools.cache
def register_pytree() -> ReexportedPyTreeModule:
    """Register Optree PyTree operations for vector objects.

    This module defines how vector objects are handled with the optree package.
    See https://blog.scientific-python.org/pytrees/ for the rationale for these functions.

    After calling this function,

    >>> import vector
    >>> vector.register_pytree()
    <module 'vector.pytree'>

    the following classes can be flattened and unflattened with the `optree` package:

    - VectorObject*D
    - MomentumObject*D
    - VectorNumpy*D
    - MomentumNumpy*D

    For example:

    >>> import optree
    >>> vec = vector.obj(x=1, y=2)
    >>> leaves, treedef = optree.tree_flatten(vec, namespace="vector")
    >>> vec2 = optree.tree_unflatten(treedef, leaves)
    >>> assert vec == vec2

    As a convenience, we return a re-exported module that can be used without the ``namespace``
    argument. For example:

    >>> pytree = vector.register_pytree()
    >>> vec = vector.obj(x=1, y=2)
    >>> leaves, treedef = pytree.flatten(vec)
    >>> vec2 = pytree.unflatten(treedef, leaves)
    >>> assert vec == vec2

    A ravel function is also added to the returned PyTree module,
    which can be used to flatten VectorNumpy arrays into a 1D array and reconstruct them.

    >>> import numpy as np
    >>> vec = vector.array({"x": np.ones(10), "y": np.ones(10)})
    >>> flat, unravel = pytree.ravel(vec)
    >>> assert flat.shape == (20,)
    >>> vec2 = unravel(flat)
    >>> assert (vec == vec2).all()

    Note that this function requires the `optree` package to be installed.
    """
    try:
        import optree.pytree
        from optree import GetAttrEntry
        from optree.integrations.numpy import tree_ravel
    except ImportError as e:
        raise ImportError("Please install optree to use vector.pytree") from e

    pytree = optree.pytree.reexport(namespace="vector", module="vector.pytree")

    pytree.register_node(
        VectorObject2D,
        flatten_func=_flatten2D,
        unflatten_func=_unflatten2D,
        path_entry_type=GetAttrEntry,
    )
    pytree.register_node(
        MomentumObject2D,
        flatten_func=_flatten2D,
        unflatten_func=_unflatten2D,
        path_entry_type=GetAttrEntry,
    )
    pytree.register_node(
        VectorObject3D,
        flatten_func=_flatten3D,
        unflatten_func=_unflatten3D,
        path_entry_type=GetAttrEntry,
    )
    pytree.register_node(
        MomentumObject3D,
        flatten_func=_flatten3D,
        unflatten_func=_unflatten3D,
        path_entry_type=GetAttrEntry,
    )
    pytree.register_node(
        VectorObject4D,
        flatten_func=_flatten4D,
        unflatten_func=_unflatten4D,
        path_entry_type=GetAttrEntry,
    )
    pytree.register_node(
        MomentumObject4D,
        flatten_func=_flatten4D,
        unflatten_func=_unflatten4D,
        path_entry_type=GetAttrEntry,
    )

    for cls in (
        VectorNumpy2D,
        MomentumNumpy2D,
        VectorNumpy3D,
        MomentumNumpy3D,
        VectorNumpy4D,
        MomentumNumpy4D,
    ):
        pytree.register_node(
            cls,
            flatten_func=_flattenAoSdata,
            unflatten_func=_unflattenAoSdata,
            path_entry_type=GetAttrEntry,
        )

    # A convenience function
    pytree.ravel = partial(tree_ravel, namespace="vector")  # type: ignore[attr-defined]

    return pytree
