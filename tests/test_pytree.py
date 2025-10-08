# Copyright (c) 2025, Nick Smith
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import numpy as np
import pytest

import vector

try:
    pytree = vector.register_pytree()
except ImportError:
    pytest.skip("optree is not installed", allow_module_level=True)


def test_pytree_roundtrip_VectorObject2D():
    vec = vector.obj(x=1, y=2)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2

    vec = vector.obj(rho=1, phi=2)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)


def test_pytree_roundtrip_MomentumObject2D():
    vec = vector.obj(px=1, py=2)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2

    vec = vector.obj(pt=1, phi=2)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)


def test_pytree_roundtrip_VectorObject3D():
    vec = vector.obj(x=1, y=2, z=3)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2, 3]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2

    vec = vector.obj(rho=1, phi=2, z=3)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2, 3]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)

    vec = vector.obj(x=1, y=2, theta=3)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2, 3]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)


def test_pytree_roundtrip_MomentumObject3D():
    vec = vector.obj(px=1, py=2, pz=3)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2, 3]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2

    vec = vector.obj(pt=1, phi=2, pz=3)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2, 3]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)

    vec = vector.obj(px=1, py=2, theta=3)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2, 3]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)


def test_pytree_roundtrip_VectorObject4D():
    vec = vector.obj(x=1, y=2, z=3, t=4)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2, 3, 4]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2

    vec = vector.obj(rho=1, phi=2, z=3, t=4)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2, 3, 4]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)
    assert type(vec.temporal) is type(vec2.temporal)

    vec = vector.obj(x=1, y=2, theta=3, t=4)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2, 3, 4]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)
    assert type(vec.temporal) is type(vec2.temporal)

    vec = vector.obj(pt=1, phi=2, eta=3, mass=4)
    leaves, treedef = pytree.flatten(vec)
    assert leaves == [1, 2, 3, 4]
    vec2 = pytree.unflatten(treedef, leaves)
    assert vec == vec2
    assert type(vec.azimuthal) is type(vec2.azimuthal)
    assert type(vec.longitudinal) is type(vec2.longitudinal)
    assert type(vec.temporal) is type(vec2.temporal)


def test_pytree_roundtrip_SoA():
    vec = vector.obj(x=1, y=1) * np.ones(10)
    flat, unravel = pytree.ravel(vec)
    assert flat.shape == (20,)
    assert (unravel(flat) == vec).all()

    vec = vector.obj(pt=1, eta=1, phi=1, mass=1) * np.ones(10)
    flat, unravel = pytree.ravel(vec)
    assert flat.shape == (40,)
    assert (unravel(flat) == vec).all()


def test_pytree_roundtrip_AoS():
    vec = vector.array(
        {
            "x": np.ones(10),
            "y": np.ones(10),
        }
    )
    flat, unravel = pytree.ravel(vec)
    assert flat.shape == (20,)
    assert flat.dtype == np.float64
    assert (unravel(flat) == vec).all()

    vec = vector.array(
        {
            "pt": np.ones(10),
            "eta": np.ones(10),
            "phi": np.ones(10),
            "mass": np.ones(10),
        }
    )
    flat, unravel = pytree.ravel(vec)
    assert flat.shape == (40,)
    assert flat.dtype == np.float64
    assert (unravel(flat) == vec).all()


def test_run_once():
    # Calling register_pytree multiple times returns the same object
    pytree2 = vector.register_pytree()
    assert pytree is pytree2
