"""PyTree operations for vector objects.

This module defines how vector objects are handled within optree.
See https://blog.scientific-python.org/pytrees/ for the rationale for these functions.
"""

from __future__ import annotations

from types import ModuleType

from vector._methods import (
    Vector2D,
    Vector3D,
    Vector4D,
)
from vector.backends.object import (
    MomentumObject2D,
    MomentumObject3D,
    MomentumObject4D,
    VectorObject2D,
    VectorObject3D,
    VectorObject4D,
)


def flatten2D(v: Vector2D) -> tuple[tuple, tuple]:
    children = v.azimuthal.elements
    metadata = type(v), type(v.azimuthal)
    return children, metadata


def unflatten2D(metadata: tuple, children: tuple) -> Vector2D:
    backend, azimuthal = metadata
    return backend(azimuthal=azimuthal(*children))


def flatten3D(v: Vector3D) -> tuple[tuple, tuple]:
    children = v.azimuthal.elements, v.longitudinal.elements
    metadata = type(v), type(v.azimuthal), type(v.longitudinal)
    return children, metadata


def unflatten3D(metadata: tuple, children: tuple) -> Vector3D:
    coords_azimuthal, coords_longitudinal = children
    backend, azimuthal, longitudinal = metadata
    return backend(
        azimuthal=azimuthal(*coords_azimuthal),
        longitudinal=longitudinal(*coords_longitudinal),
    )


def flatten4D(v: Vector4D) -> tuple[tuple, tuple]:
    children = (
        v.azimuthal.elements,
        v.longitudinal.elements,
        v.temporal.elements,
    )
    metadata = type(v), type(v.azimuthal), type(v.longitudinal), type(v.temporal)
    return children, metadata


def unflatten4D(metadata: tuple, children: tuple) -> Vector4D:
    coords_azimuthal, coords_longitudinal, coords_temporal = children
    backend, azimuthal, longitudinal, temporal = metadata
    return backend(
        azimuthal=azimuthal(*coords_azimuthal),
        longitudinal=longitudinal(*coords_longitudinal),
        temporal=temporal(*coords_temporal),
    )


def _register(pytree: ModuleType) -> None:
    """Register vector objects with the given pytree module."""

    pytree.register_node(
        VectorObject2D, flatten_func=flatten2D, unflatten_func=unflatten2D
    )
    pytree.register_node(
        MomentumObject2D, flatten_func=flatten2D, unflatten_func=unflatten2D
    )
    pytree.register_node(
        VectorObject3D, flatten_func=flatten3D, unflatten_func=unflatten3D
    )
    pytree.register_node(
        MomentumObject3D, flatten_func=flatten3D, unflatten_func=unflatten3D
    )
    pytree.register_node(
        VectorObject4D, flatten_func=flatten4D, unflatten_func=unflatten4D
    )
    pytree.register_node(
        MomentumObject4D, flatten_func=flatten4D, unflatten_func=unflatten4D
    )
