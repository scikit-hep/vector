# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import itertools

import numpy as np
import pytest

import vector
from vector._methods import Momentum

ak = pytest.importorskip("awkward")
sympy = pytest.importorskip("sympy")
numba = pytest.importorskip("numba")

pytestmark = [pytest.mark.awkward, pytest.mark.sympy, pytest.mark.numba]


ALL_COORDINATES = [
    "x",
    "y",
    "rho",
    "phi",
    "px",
    "py",
    "pt",
    "z",
    "theta",
    "eta",
    "pz",
    "t",
    "tau",
    "E",
    "e",
    "energy",
    "M",
    "m",
    "mass",
]

MOMENTUM_COORDINATES = {"px", "py", "pz", "pt", "E", "e", "energy", "M", "m", "mass"}
AZIMUTHAL_COORDS = {"x", "y", "rho", "phi", "px", "py", "pt"}
TEMPORAL_COORDS = {"t", "tau", "E", "e", "energy", "M", "m", "mass"}
LONGITUDINAL_COORDS = {"z", "theta", "eta", "pz"}

COORD_ALIASES = {
    "px": "x",
    "py": "y",
    "pz": "z",
    "pt": "rho",
    "E": "t",
    "e": "t",
    "energy": "t",
    "M": "tau",
    "m": "tau",
    "mass": "tau",
}

VALID_2_COMBINATIONS = [
    {"x", "y"},
    {"rho", "phi"},
    {"px", "py"},
    {"pt", "phi"},
    {"x", "py"},
    {"px", "y"},
]

ALL_2_COMBINATIONS = list(itertools.combinations(ALL_COORDINATES, 2))

VALID_3_COMBINATIONS = [
    az | {lon} for az in VALID_2_COMBINATIONS for lon in ("z", "theta", "eta", "pz")
]

ALL_3_COMBINATIONS = list(itertools.combinations(ALL_COORDINATES, 3))

VALID_4_COMBINATIONS = [
    az | {lon} | {temp}
    for az in VALID_2_COMBINATIONS
    for lon in ("z", "theta", "eta", "pz")
    for temp in ("t", "tau", "E", "e", "energy", "M", "m", "mass")
]

ALL_4_COMBINATIONS = list(itertools.combinations(ALL_COORDINATES, 4))


def _is_valid_2(combo):
    return set(combo) in VALID_2_COMBINATIONS


def _is_valid_3(combo):
    return set(combo) in VALID_3_COMBINATIONS


def _is_valid_4(combo):
    return set(combo) in VALID_4_COMBINATIONS


def _has_valid_3_subset(combo):
    for triple in itertools.combinations(combo, 3):
        if set(triple) in VALID_3_COMBINATIONS:
            return True
    return False


def _has_valid_2_subset(combo):
    for pair in itertools.combinations(combo, 2):
        if set(pair) in VALID_2_COMBINATIONS:
            return True
    return False


def _is_momentum(combo):
    return any(c in MOMENTUM_COORDINATES for c in combo)


def _to_canonical(coord):
    return COORD_ALIASES.get(coord, coord)


def _has_duplicate(combo):
    canonicals = [_to_canonical(c) for c in combo]
    return len(canonicals) != len(set(canonicals))


def _will_error_for_non_obj(combo):
    """Check if combo with valid subset will error for non-obj backends."""
    # Check for canonical duplicates (e.g., x and px both map to x)
    if _has_duplicate(combo):
        return True

    # Temporal without longitudinal
    has_temporal = any(c in TEMPORAL_COORDS for c in combo)
    has_longitudinal = any(c in LONGITUDINAL_COORDS for c in combo)
    if has_temporal and not has_longitudinal:
        return True

    # Multiple longitudinal coords (z, theta, eta, pz) - only one allowed
    longitudinal_count = sum(1 for c in combo if c in LONGITUDINAL_COORDS)
    if longitudinal_count > 1:
        return True

    # Multiple temporal coords - only one allowed
    temporal_count = sum(1 for c in combo if c in TEMPORAL_COORDS)
    if temporal_count > 1:
        return True

    # Multiple azimuthal pairs (e.g., x,y and rho,phi both present)
    valid_2_count = sum(
        1
        for pair in itertools.combinations(combo, 2)
        if set(pair) in VALID_2_COMBINATIONS
    )
    return valid_2_count > 1


def _get_first_valid_2_subset(combo):
    """Get the first valid 2-subset from a combo."""
    for pair in itertools.combinations(combo, 2):
        if set(pair) in VALID_2_COMBINATIONS:
            return set(pair)
    return None


def _get_first_valid_3_subset(combo):
    """Get the first valid 3-subset from a combo."""
    for triple in itertools.combinations(combo, 3):
        if set(triple) in VALID_3_COMBINATIONS:
            return set(triple)
    return None


def _is_momentum_numpy(combo):
    """Numpy checks if ANY field name is a momentum coordinate."""
    return any(c in MOMENTUM_COORDINATES for c in combo)


def _is_momentum_awkward(combo):
    """Awkward only sets is_momentum when momentum coords are consumed as vector coords."""
    # Check for valid 4-subset first
    for quad in itertools.combinations(combo, 4):
        if set(quad) in VALID_4_COMBINATIONS:
            return _is_momentum(quad)
    # Check for valid 3-subset
    valid_3 = _get_first_valid_3_subset(combo)
    if valid_3 is not None:
        return _is_momentum(valid_3)
    # Check for valid 2-subset
    valid_2 = _get_first_valid_2_subset(combo)
    if valid_2 is not None:
        return _is_momentum(valid_2)
    return False


def _get_sympy_class(coords):
    has_momentum = _is_momentum(coords)
    n = len(coords)
    if n <= 2:
        return vector.MomentumSympy2D if has_momentum else vector.VectorSympy2D
    elif n == 3:
        return vector.MomentumSympy3D if has_momentum else vector.VectorSympy3D
    else:
        return vector.MomentumSympy4D if has_momentum else vector.VectorSympy4D


def _numba_obj(combo):
    """Create vector.obj inside a jitted function with the given coordinates."""
    kwargs = ", ".join(f"{c}=1.0" for c in combo)
    local_ns = {"vector": vector, "numba": numba}
    exec(f"@numba.njit\ndef f():\n    return vector.obj({kwargs})", local_ns)
    return local_ns["f"]


def _will_numba_error(combo):
    """Check if numba will error. Numba errors on duplicates, no valid azimuthal, or extra azimuthal coords."""
    if _has_duplicate(combo):
        return True
    if not _has_valid_2_subset(combo):
        return True
    # Numba errors if there are more than 2 azimuthal coordinates (canonical form)
    canonical_azimuthal = {_to_canonical(c) for c in combo if c in AZIMUTHAL_COORDS}
    return len(canonical_azimuthal) > 2


@pytest.mark.parametrize(
    "combo",
    ALL_2_COMBINATIONS,
    ids=[f"{a}_{b}" for a, b in ALL_2_COMBINATIONS],
)
def test_2_combinations(combo):
    is_valid = _is_valid_2(combo)
    is_momentum = _is_momentum(combo)
    error_pattern = "duplicate coordinates|unrecognized combination|must have a structured dtype|specify"

    numba_error_pattern = "duplicate coordinates|unrecognized combination"

    if is_valid:
        v_obj = vector.obj(**dict.fromkeys(combo, 1.0))
        v_numba = _numba_obj(combo)()
        v_numpy = vector.array({c: np.array([1.0, 2.0]) for c in combo})
        v_awkward = vector.Array(ak.Array({c: [1.0, 2.0] for c in combo}))
        v_zip = vector.zip({c: np.array([1.0, 2.0]) for c in combo})
        v_sympy = _get_sympy_class(combo)(**{c: sympy.Symbol(c) for c in combo})

        assert isinstance(v_obj, Momentum) == is_momentum
        assert isinstance(v_numba, Momentum) == is_momentum
        assert isinstance(v_numpy, Momentum) == is_momentum
        assert isinstance(v_awkward, Momentum) == is_momentum
        assert isinstance(v_zip, Momentum) == is_momentum
        assert isinstance(v_sympy, Momentum) == is_momentum
    else:
        with pytest.raises(TypeError, match=error_pattern):
            vector.obj(**dict.fromkeys(combo, 1.0))

        with pytest.raises(numba.TypingError, match=numba_error_pattern):
            _numba_obj(combo)()

        with pytest.raises(TypeError, match=error_pattern):
            vector.array({c: np.array([1.0, 2.0]) for c in combo})

        with pytest.raises(TypeError, match=error_pattern):
            vector.Array(ak.Array({c: [1.0, 2.0] for c in combo}))

        with pytest.raises(TypeError, match=error_pattern):
            vector.zip({c: np.array([1.0, 2.0]) for c in combo})

        with pytest.raises(TypeError, match=error_pattern):
            _get_sympy_class(combo)(**{c: sympy.Symbol(c) for c in combo})


@pytest.mark.parametrize(
    "combo",
    ALL_3_COMBINATIONS,
    ids=[f"{a}_{b}_{c}" for a, b, c in ALL_3_COMBINATIONS],
)
def test_3_combinations(combo):
    is_valid = _is_valid_3(combo)
    has_valid_2 = _has_valid_2_subset(combo)
    is_momentum = _is_momentum(combo)
    error_pattern = "duplicate coordinates|unrecognized combination|must have a structured dtype|specify"
    numba_error_pattern = "duplicate coordinates|unrecognized combination"

    if is_valid:
        v_obj = vector.obj(**dict.fromkeys(combo, 1.0))
        v_numba = _numba_obj(combo)()
        v_numpy = vector.array({c: np.array([1.0, 2.0]) for c in combo})
        v_awkward = vector.Array(ak.Array({c: [1.0, 2.0] for c in combo}))
        v_zip = vector.zip({c: np.array([1.0, 2.0]) for c in combo})
        v_sympy = _get_sympy_class(combo)(**{c: sympy.Symbol(c) for c in combo})

        assert isinstance(v_obj, Momentum) == is_momentum
        assert isinstance(v_numba, Momentum) == is_momentum
        assert isinstance(v_numpy, Momentum) == is_momentum
        assert isinstance(v_awkward, Momentum) == is_momentum
        assert isinstance(v_zip, Momentum) == is_momentum
        assert isinstance(v_sympy, Momentum) == is_momentum
    else:
        # obj and sympy are strict - always error for invalid combos
        with pytest.raises(TypeError, match=error_pattern):
            vector.obj(**dict.fromkeys(combo, 1.0))

        with pytest.raises(TypeError, match=error_pattern):
            _get_sympy_class(combo)(**{c: sympy.Symbol(c) for c in combo})

        # numba is permissive like numpy/awkward/zip
        if _will_numba_error(combo):
            with pytest.raises(numba.TypingError, match=numba_error_pattern):
                _numba_obj(combo)()
        else:
            _numba_obj(combo)()

        if has_valid_2 and not _will_error_for_non_obj(combo):
            # numpy/awkward/zip create a 2D vector with extra fields
            v_numpy = vector.array({c: np.array([1.0, 2.0]) for c in combo})
            v_awkward = vector.Array(ak.Array({c: [1.0, 2.0] for c in combo}))
            v_zip = vector.zip({c: np.array([1.0, 2.0]) for c in combo})

            assert isinstance(v_numpy, Momentum) == _is_momentum_numpy(combo)
            assert isinstance(v_awkward, Momentum) == _is_momentum_awkward(combo)
            assert isinstance(v_zip, Momentum) == _is_momentum_awkward(combo)
        else:
            with pytest.raises(TypeError, match=error_pattern):
                vector.array({c: np.array([1.0, 2.0]) for c in combo})

            with pytest.raises(TypeError, match=error_pattern):
                vector.Array(ak.Array({c: [1.0, 2.0] for c in combo}))

            with pytest.raises(TypeError, match=error_pattern):
                vector.zip({c: np.array([1.0, 2.0]) for c in combo})


@pytest.mark.parametrize(
    "combo",
    ALL_4_COMBINATIONS,
    ids=[f"{a}_{b}_{c}_{d}" for a, b, c, d in ALL_4_COMBINATIONS],
)
def test_4_combinations(combo):
    is_valid = _is_valid_4(combo)
    has_valid_3 = _has_valid_3_subset(combo)
    has_valid_2 = _has_valid_2_subset(combo)
    is_momentum = _is_momentum(combo)
    error_pattern = "duplicate coordinates|unrecognized combination|must have a structured dtype|specify"
    numba_error_pattern = "duplicate coordinates|unrecognized combination"

    if is_valid:
        v_obj = vector.obj(**dict.fromkeys(combo, 1.0))
        v_numba = _numba_obj(combo)()
        v_numpy = vector.array({c: np.array([1.0, 2.0]) for c in combo})
        v_awkward = vector.Array(ak.Array({c: [1.0, 2.0] for c in combo}))
        v_zip = vector.zip({c: np.array([1.0, 2.0]) for c in combo})
        v_sympy = _get_sympy_class(combo)(**{c: sympy.Symbol(c) for c in combo})

        assert isinstance(v_obj, Momentum) == is_momentum
        assert isinstance(v_numba, Momentum) == is_momentum
        assert isinstance(v_numpy, Momentum) == is_momentum
        assert isinstance(v_awkward, Momentum) == is_momentum
        assert isinstance(v_zip, Momentum) == is_momentum
        assert isinstance(v_sympy, Momentum) == is_momentum
    else:
        # obj and sympy are strict - always error for invalid combos
        with pytest.raises(TypeError, match=error_pattern):
            vector.obj(**dict.fromkeys(combo, 1.0))

        with pytest.raises(TypeError, match=error_pattern):
            _get_sympy_class(combo)(**{c: sympy.Symbol(c) for c in combo})

        # numba is permissive like numpy/awkward/zip
        if _will_numba_error(combo):
            with pytest.raises(numba.TypingError, match=numba_error_pattern):
                _numba_obj(combo)()
        else:
            _numba_obj(combo)()

        if has_valid_3 and not _will_error_for_non_obj(combo):
            # numpy/awkward/zip create a 3D vector with extra fields
            v_numpy = vector.array({c: np.array([1.0, 2.0]) for c in combo})
            v_awkward = vector.Array(ak.Array({c: [1.0, 2.0] for c in combo}))
            v_zip = vector.zip({c: np.array([1.0, 2.0]) for c in combo})

            assert isinstance(v_numpy, Momentum) == _is_momentum_numpy(combo)
            assert isinstance(v_awkward, Momentum) == _is_momentum_awkward(combo)
            assert isinstance(v_zip, Momentum) == _is_momentum_awkward(combo)
        elif has_valid_2 and not _will_error_for_non_obj(combo):
            # numpy/awkward/zip create a 2D vector with extra fields
            v_numpy = vector.array({c: np.array([1.0, 2.0]) for c in combo})
            v_awkward = vector.Array(ak.Array({c: [1.0, 2.0] for c in combo}))
            v_zip = vector.zip({c: np.array([1.0, 2.0]) for c in combo})

            assert isinstance(v_numpy, Momentum) == _is_momentum_numpy(combo)
            assert isinstance(v_awkward, Momentum) == _is_momentum_awkward(combo)
            assert isinstance(v_zip, Momentum) == _is_momentum_awkward(combo)
        else:
            with pytest.raises(TypeError, match=error_pattern):
                vector.array({c: np.array([1.0, 2.0]) for c in combo})

            with pytest.raises(TypeError, match=error_pattern):
                vector.Array(ak.Array({c: [1.0, 2.0] for c in combo}))

            with pytest.raises(TypeError, match=error_pattern):
                vector.zip({c: np.array([1.0, 2.0]) for c in combo})
