# Copyright (c) 2025, Nick Smith
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import pytest

import vector

if vector.pytree is None:
    pytest.skip("optree is not installed", allow_module_level=True)


def test_pytree_roundtrip_VectorObject2D():
    vec = vector.obj(x=1, y=2)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2

    vec = vector.obj(rho=1, phi=2)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)


def test_pytree_roundtrip_MomentumObject2D():
    vec = vector.obj(px=1, py=2)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2

    vec = vector.obj(pt=1, phi=2)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)


def test_pytree_roundtrip_VectorObject3D():
    vec = vector.obj(x=1, y=2, z=3)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2, 3]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2

    vec = vector.obj(rho=1, phi=2, z=3)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2, 3]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)

    vec = vector.obj(x=1, y=2, theta=3)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2, 3]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)


def test_pytree_roundtrip_MomentumObject3D():
    vec = vector.obj(px=1, py=2, pz=3)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2, 3]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2

    vec = vector.obj(pt=1, phi=2, pz=3)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2, 3]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)

    vec = vector.obj(px=1, py=2, theta=3)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2, 3]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)


def test_pytree_roundtrip_VectorObject4D():
    vec = vector.obj(x=1, y=2, z=3, t=4)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2, 3, 4]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2

    vec = vector.obj(rho=1, phi=2, z=3, t=4)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2, 3, 4]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)
    assert type(vec.temporal) is type(vec2.temporal)

    vec = vector.obj(x=1, y=2, theta=3, t=4)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2, 3, 4]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)
    assert type(vec.temporal) is type(vec2.temporal)

    vec = vector.obj(pt=1, phi=2, eta=3, mass=4)
    leaves, treedef = vector.pytree.flatten(vec)
    assert leaves == [1, 2, 3, 4]
    vec2 = vector.pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)
    assert type(vec.temporal) is type(vec2.temporal)
