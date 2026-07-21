# Copyright (c) 2019, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
Exhaustive tests of the coordinate names that every backend accepts, added with
https://github.com/scikit-hep/vector/pull/659.

The rules are implemented once, in ``vector._methods._check_coordinate_names``,
so the tests below check every backend against the same independently written
reference implementation (:func:`reference`) and against each other.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import vector

# Momentum-aliases; ``phi``, ``theta``, and ``eta`` have none.
ALIASES = {
    "px": "x",
    "py": "y",
    "pt": "rho",
    "pz": "z",
    "E": "t",
    "e": "t",
    "energy": "t",
    "M": "tau",
    "m": "tau",
    "mass": "tau",
}

GENERIC_NAMES = ("x", "y", "rho", "phi", "z", "theta", "eta", "t", "tau")

ALIAS_CHOICES = {
    "x": ("x", "px"),
    "y": ("y", "py"),
    "rho": ("rho", "pt"),
    "phi": ("phi",),
    "z": ("z", "pz"),
    "theta": ("theta",),
    "eta": ("eta",),
    "t": ("t", "E", "e", "energy"),
    "tau": ("tau", "M", "m", "mass"),
}


def reference(names: tuple[str, ...]) -> tuple[bool, int] | None:
    """
    Independent implementation of the rules: returns ``(is_momentum, dimension)``
    for a combination of coordinate names that describes exactly one vector and
    None for one that does not.
    """
    generic = [ALIASES.get(name, name) for name in names]
    if len(set(generic)) != len(generic):
        return None

    remaining = set(generic)
    for azimuthal in ({"x", "y"}, {"rho", "phi"}):
        if azimuthal <= remaining:
            remaining -= azimuthal
            break
    else:
        return None

    dimension = 2
    for longitudinal in ("z", "theta", "eta"):
        if longitudinal in remaining:
            remaining.remove(longitudinal)
            dimension = 3
            break
    if dimension == 3:
        for temporal in ("t", "tau"):
            if temporal in remaining:
                remaining.remove(temporal)
                dimension = 4
                break

    if remaining:
        return None

    return any(name in ALIASES for name in names), dimension


def is_valid_for(
    names: tuple[str, ...], dimension: int | None = None, momentum: bool | None = None
) -> bool:
    """
    Whether ``names`` may be given to a constructor that is restricted to a
    ``dimension`` and/or to (non-)momentum coordinates.
    """
    expected = reference(names)
    if expected is None:
        return False
    is_momentum, expected_dimension = expected
    if dimension is not None and dimension != expected_dimension:
        return False
    return not (is_momentum and momentum is False)


# All 511 non-empty subsets of the generic coordinate names, of which 20 are vectors.
GENERIC_COMBINATIONS = [
    names
    for size in range(1, len(GENERIC_NAMES) + 1)
    for names in itertools.combinations(GENERIC_NAMES, size)
]

VECTOR_COMBINATIONS = [names for names in GENERIC_COMBINATIONS if reference(names)]

# The 6 + 24 + 192 combinations of coordinate names, including momentum-aliases.
ALIAS_COMBINATIONS = [
    names
    for generic in VECTOR_COMBINATIONS
    for names in itertools.product(*(ALIAS_CHOICES[name] for name in generic))
]


def values(names: tuple[str, ...]) -> dict[str, float]:
    """A distinct value for each coordinate, so that mix-ups are visible."""
    return {name: 1.0 + i for i, name in enumerate(names)}


def test_combinations_are_exhaustive():
    assert len(GENERIC_COMBINATIONS) == 511
    assert len(VECTOR_COMBINATIONS) == 2 + 6 + 12
    assert len(ALIAS_COMBINATIONS) == 6 + 24 + 192


object_classes = {
    (2, False): vector.VectorObject2D,
    (3, False): vector.VectorObject3D,
    (4, False): vector.VectorObject4D,
    (2, True): vector.MomentumObject2D,
    (3, True): vector.MomentumObject3D,
    (4, True): vector.MomentumObject4D,
}

numpy_classes = {
    (2, False): vector.VectorNumpy2D,
    (3, False): vector.VectorNumpy3D,
    (4, False): vector.VectorNumpy4D,
    (2, True): vector.MomentumNumpy2D,
    (3, True): vector.MomentumNumpy3D,
    (4, True): vector.MomentumNumpy4D,
}


def numpy_array(coordinates: dict[str, float]) -> np.ndarray:
    return np.array(
        [tuple(coordinates.values())],
        dtype=[(name, np.float64) for name in coordinates],
    )


def record_name(names: tuple[str, ...]) -> str:
    is_momentum, dimension = reference(names)
    return f"{'Momentum' if is_momentum else 'Vector'}{dimension}D"


@pytest.mark.parametrize("names", GENERIC_COMBINATIONS)
def test_obj_generic_combinations(names):
    coordinates = values(names)

    if reference(names) is None:
        with pytest.raises(TypeError):
            vector.obj(**coordinates)
    else:
        vec = vector.obj(**coordinates)
        assert isinstance(vec, object_classes[reference(names)[1], False])
        for name, value in coordinates.items():
            assert getattr(vec, name) == pytest.approx(value)


@pytest.mark.parametrize("names", GENERIC_COMBINATIONS)
def test_object_class_generic_combinations(names):
    coordinates = values(names)

    for (dimension, momentum), cls in object_classes.items():
        if is_valid_for(names, dimension, momentum):
            vec = cls(**coordinates)
            for name, value in coordinates.items():
                assert getattr(vec, name) == pytest.approx(value)
        else:
            with pytest.raises(TypeError):
                cls(**coordinates)


@pytest.mark.parametrize("names", GENERIC_COMBINATIONS)
def test_array_generic_combinations(names):
    coordinates = values(names)

    if reference(names) is None:
        with pytest.raises(TypeError):
            vector.array(
                {name: np.array([value]) for name, value in coordinates.items()}
            )
    else:
        vec = vector.array(
            {name: np.array([value]) for name, value in coordinates.items()}
        )
        assert isinstance(vec, numpy_classes[reference(names)[1], False])
        for name, value in coordinates.items():
            assert getattr(vec, name)[0] == pytest.approx(value)


@pytest.mark.parametrize("names", GENERIC_COMBINATIONS)
def test_numpy_class_generic_combinations(names):
    array = numpy_array(values(names))

    for (dimension, momentum), cls in numpy_classes.items():
        if is_valid_for(names, dimension, momentum):
            vec = array.view(cls)
            for name, value in values(names).items():
                assert vec[name][0] == pytest.approx(value)
        else:
            with pytest.raises(TypeError):
                array.view(cls)


@pytest.mark.parametrize("names", GENERIC_COMBINATIONS)
def test_awkward_generic_combinations(names):
    ak = pytest.importorskip("awkward")
    coordinates = values(names)
    columns = {name: [value] for name, value in coordinates.items()}

    if reference(names) is None:
        with pytest.raises(TypeError):
            vector.Array(ak.Array(columns))
        with pytest.raises(TypeError):
            vector.zip(columns)
    else:
        for vec in (vector.Array(ak.Array(columns)), vector.zip(columns)):
            assert vec.layout.purelist_parameter("__record__") == record_name(names)
            for name, value in coordinates.items():
                assert getattr(vec, name)[0] == pytest.approx(value)


@pytest.mark.parametrize("names", GENERIC_COMBINATIONS)
def test_sympy_generic_combinations(names):
    sympy = pytest.importorskip("sympy")
    coordinates = {name: sympy.Symbol(name) for name in names}

    sympy_classes = {
        (2, False): vector.VectorSympy2D,
        (3, False): vector.VectorSympy3D,
        (4, False): vector.VectorSympy4D,
        (2, True): vector.MomentumSympy2D,
        (3, True): vector.MomentumSympy3D,
        (4, True): vector.MomentumSympy4D,
    }

    for (dimension, momentum), cls in sympy_classes.items():
        if is_valid_for(names, dimension, momentum):
            vec = cls(**coordinates)
            for name, symbol in coordinates.items():
                assert getattr(vec, name) == symbol
        else:
            with pytest.raises(TypeError):
                cls(**coordinates)


@pytest.mark.parametrize("names", ALIAS_COMBINATIONS)
def test_alias_combinations(names):
    is_momentum, dimension = reference(names)
    coordinates = values(names)

    vec = vector.obj(**coordinates)
    assert isinstance(vec, object_classes[dimension, is_momentum])

    array = vector.array(
        {name: np.array([value]) for name, value in coordinates.items()}
    )
    assert isinstance(array, numpy_classes[dimension, is_momentum])

    for name, value in coordinates.items():
        assert getattr(vec, name) == pytest.approx(value)
        assert getattr(array, name)[0] == pytest.approx(value)
        # the generic name of an alias reads back the same coordinate
        generic = ALIASES.get(name, name)
        assert getattr(vec, generic) == pytest.approx(value)
        assert getattr(array, generic)[0] == pytest.approx(value)


@pytest.mark.parametrize("names", ALIAS_COMBINATIONS)
def test_alias_combinations_awkward(names):
    ak = pytest.importorskip("awkward")
    coordinates = values(names)
    columns = {name: [value] for name, value in coordinates.items()}

    for vec in (vector.Array(ak.Array(columns)), vector.zip(columns)):
        assert vec.layout.purelist_parameter("__record__") == record_name(names)
        for name, value in coordinates.items():
            assert getattr(vec, name)[0] == pytest.approx(value)


@pytest.mark.parametrize("names", ALIAS_COMBINATIONS)
def test_duplicate_aliases(names):
    """Adding any name that maps to a coordinate already given is an error."""
    coordinates = values(names)

    for name in names:
        for alias in ALIAS_CHOICES[ALIASES.get(name, name)]:
            if alias == name:
                continue
            duplicated = {**coordinates, alias: 99.0}
            with pytest.raises(TypeError, match="duplicate coordinates"):
                vector.obj(**duplicated)
            with pytest.raises(TypeError, match="duplicate coordinates"):
                vector.array(
                    {k: np.array([v]) for k, v in duplicated.items()},
                )


def test_duplicate_aliases_awkward():
    ak = pytest.importorskip("awkward")

    for names in ALIAS_COMBINATIONS:
        coordinates = values(names)
        for name in names:
            for alias in ALIAS_CHOICES[ALIASES.get(name, name)]:
                if alias == name:
                    continue
                duplicated = {**coordinates, alias: 99.0}
                columns = {k: [v] for k, v in duplicated.items()}
                with pytest.raises(TypeError, match="duplicate coordinates"):
                    vector.Array(ak.Array(columns))
                with pytest.raises(TypeError, match="duplicate coordinates"):
                    vector.zip(columns)


def test_generic_vectors_reject_momentum_aliases():
    with pytest.raises(TypeError, match="momentum-aliases are not allowed"):
        vector.VectorObject2D(px=1.0, py=2.0)
    with pytest.raises(TypeError, match="momentum-aliases are not allowed"):
        vector.VectorObject3D(x=1.0, y=2.0, pz=3.0)
    with pytest.raises(TypeError, match="momentum-aliases are not allowed"):
        vector.VectorObject4D(x=1.0, y=2.0, z=3.0, energy=4.0)
    with pytest.raises(TypeError, match="momentum-aliases are not allowed"):
        numpy_array({"px": 1.0, "py": 2.0}).view(vector.VectorNumpy2D)


def test_momentum_vectors_accept_generic_names():
    assert vector.MomentumObject2D(x=1.0, y=2.0).px == pytest.approx(1.0)
    assert vector.MomentumObject3D(x=1.0, y=2.0, z=3.0).pz == pytest.approx(3.0)
    momentum = vector.MomentumObject4D(x=1.0, y=2.0, z=3.0, t=4.0)
    assert momentum.energy == pytest.approx(4.0)


def test_conflicting_coordinates():
    with pytest.raises(TypeError, match="specify x= and y= or rho= and phi="):
        vector.obj(x=1.0, y=2.0, rho=3.0, phi=4.0)
    with pytest.raises(TypeError, match="specify z= or theta= or eta="):
        vector.obj(x=1.0, y=2.0, z=3.0, eta=4.0)
    with pytest.raises(TypeError, match="specify t= or tau="):
        vector.obj(x=1.0, y=2.0, z=3.0, t=4.0, tau=5.0)
    with pytest.raises(TypeError, match="specify t= or tau="):
        vector.obj(pt=1.0, phi=2.0, eta=3.0, mass=4.0, energy=5.0)


def test_extra_fields():
    """Non-coordinate fields are records' payload; keyword arguments are not."""
    array = vector.array(
        {"x": np.array([1.0]), "y": np.array([2.0]), "wow": np.array([3.0])}
    )
    assert array["wow"][0] == pytest.approx(3.0)

    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(x=1.0, y=2.0, wow=3.0)
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.VectorObject2D(x=1.0, y=2.0, wow=3.0)


def test_extra_fields_awkward():
    ak = pytest.importorskip("awkward")

    for vec in (
        vector.Array(ak.Array({"x": [1.0], "y": [2.0], "wow": [3.0]})),
        vector.zip({"x": [1.0], "y": [2.0], "wow": [3.0]}),
    ):
        assert vec.wow[0] == pytest.approx(3.0)


def test_unstructured_numpy_array():
    with pytest.raises(TypeError, match="must have a structured dtype"):
        np.array([1.0, 2.0]).view(vector.VectorNumpy2D)
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.array([1.0, 2.0])


def test_coordinate_objects_are_still_required():
    with pytest.raises(TypeError, match="must give Azimuthal"):
        vector.VectorObject2D()
    with pytest.raises(TypeError, match="must give Azimuthal and Longitudinal"):
        vector.VectorObject3D()
    with pytest.raises(
        TypeError, match="must give Azimuthal, Longitudinal, and Temporal"
    ):
        vector.VectorObject4D()


def test_complaint_lists_the_allowed_combinations():
    with pytest.raises(TypeError) as excinfo:
        vector.obj(x=1.0)
    complaint = str(excinfo.value)
    for names in VECTOR_COMBINATIONS:
        assert (
            "    ({}D) {}".format(len(names), " ".join(f"{x}=" for x in names))
            in complaint
        )
    assert "or their momentum equivalents" in complaint

    # a constructor that is restricted to one dimension only lists that dimension
    with pytest.raises(TypeError) as excinfo:
        vector.VectorObject2D(x=1.0, y=2.0, z=3.0)
    complaint = str(excinfo.value)
    assert "    x= y=" in complaint
    assert "    rho= phi=" in complaint
    assert "z=" not in complaint
    assert "or their momentum equivalents" not in complaint


def awkward_validates() -> bool:
    """Whether the installed Awkward Array calls ``__awkward_validation__``."""
    ak = pytest.importorskip("awkward")

    validated = []

    class Probe(ak.Array):  # type: ignore[misc]
        def __awkward_validation__(self) -> None:
            validated.append(None)

    ak.Array(
        [{"x": 1.1}],
        behavior={("*", "probe"): Probe},
        with_name="probe",
    )
    return bool(validated)


@pytest.mark.parametrize("names", ALIAS_COMBINATIONS)
def test_awkward_behavior_validation(names):
    ak = pytest.importorskip("awkward")
    if not awkward_validates():
        pytest.skip("awkward is too old to validate behaviors")

    behavior = vector.backends.awkward.behavior
    coordinates = values(names)
    columns = {name: [value] for name, value in coordinates.items()}

    vec = ak.zip(columns, with_name=record_name(names), behavior=behavior)
    for name, value in coordinates.items():
        assert getattr(vec, name)[0] == pytest.approx(value)

    # records are validated in the same way as arrays
    assert getattr(vec[0], names[0]) == pytest.approx(coordinates[names[0]])

    # a record name that does not describe these coordinates is rejected
    for dimension in (2, 3, 4):
        for momentum in (False, True):
            name = f"{'Momentum' if momentum else 'Vector'}{dimension}D"
            if is_valid_for(names, dimension, momentum):
                ak.zip(columns, with_name=name, behavior=behavior)
            else:
                with pytest.raises(TypeError):
                    ak.zip(columns, with_name=name, behavior=behavior)


def test_awkward_behavior_validation_extra_fields():
    ak = pytest.importorskip("awkward")
    if not awkward_validates():
        pytest.skip("awkward is too old to validate behaviors")

    behavior = vector.backends.awkward.behavior
    columns = {"pt": [1.0], "phi": [2.0], "eta": [3.0], "mass": [4.0], "charge": [1]}
    vec = ak.zip(columns, with_name="Momentum4D", behavior=behavior)
    assert vec.pt[0] == pytest.approx(1.0)
    assert vec.charge[0] == 1

    # ... but a field that is a coordinate under another name is not payload
    with pytest.raises(TypeError, match="specify t= or tau="):
        ak.zip({**columns, "energy": [5.0]}, with_name="Momentum4D", behavior=behavior)
    with pytest.raises(TypeError, match="duplicate coordinates"):
        ak.zip({**columns, "rho": [5.0]}, with_name="Momentum4D", behavior=behavior)

    # assigning such a field to an existing array is caught as well
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vec["rho"] = np.array([5.0])


def test_awkward_behavior_validation_names_the_array():
    """The record name is chosen elsewhere, so the complaint has to identify it."""
    ak = pytest.importorskip("awkward")
    if not awkward_validates():
        pytest.skip("awkward is too old to validate behaviors")

    behavior = vector.backends.awkward.behavior
    columns = {"pt": [1.0], "phi": [2.0]}

    with pytest.raises(TypeError, match=r"MomentumArray4D with fields \['pt', 'phi'\]"):
        ak.zip(columns, with_name="Momentum4D", behavior=behavior)

    vec = ak.zip(columns, with_name="Momentum2D", behavior=behavior)
    with pytest.raises(TypeError, match=r"MomentumArray2D with fields"):
        del vec["phi"]


def test_awkward_behavior_validation_subclass():
    ak = pytest.importorskip("awkward")
    if not awkward_validates():
        pytest.skip("awkward is too old to validate behaviors")

    behavior = dict(vector.backends.awkward.behavior)

    class PtEtaPhiMArray(vector.backends.awkward.MomentumArray4D):  # type: ignore[misc]
        pass

    behavior["*", "PtEtaPhiM"] = PtEtaPhiMArray

    vec = ak.zip(
        {"pt": [1.0], "phi": [2.0], "eta": [3.0], "mass": [4.0]},
        with_name="PtEtaPhiM",
        behavior=behavior,
    )
    assert vec.pt[0] == pytest.approx(1.0)

    with pytest.raises(TypeError, match="specify t= or tau="):
        ak.zip(
            {"pt": [1.0], "phi": [2.0], "eta": [3.0], "mass": [4.0], "energy": [5.0]},
            with_name="PtEtaPhiM",
            behavior=behavior,
        )


# Every combination of generic names, plus enough momentum-aliases to reach every
# entry of the tables that the Numba implementation of ``vector.obj`` dispatches on.
NUMBA_COMBINATIONS = [
    *VECTOR_COMBINATIONS,
    ("px", "py"),
    ("x", "py"),
    ("px", "y"),
    ("pt", "phi"),
    ("px", "py", "pz"),
    ("px", "py", "pz", "E"),
    ("px", "py", "pz", "e"),
    ("px", "py", "pz", "energy"),
    ("px", "py", "pz", "M"),
    ("px", "py", "pz", "m"),
    ("px", "py", "pz", "mass"),
]


@pytest.mark.numba
def test_numba_obj_combinations():
    numba = pytest.importorskip("numba")
    pytest.importorskip("vector.backends._numba_object")

    # every call site is typed separately, so they are compiled together to
    # keep the (considerable) compilation time down
    source = "def make_all():\n    return (\n"
    for names in NUMBA_COMBINATIONS:
        arguments = ", ".join(
            f"{name}={value}" for name, value in values(names).items()
        )
        source += f"        vector.obj({arguments}),\n"
    source += "    )\n"

    namespace = {"vector": vector}
    exec(source, namespace)

    vectors = numba.njit(namespace["make_all"])()

    for names, vec in zip(NUMBA_COMBINATIONS, vectors, strict=True):
        is_momentum, dimension = reference(names)
        assert isinstance(vec, object_classes[dimension, is_momentum])
        for name, value in values(names).items():
            assert getattr(vec, name) == pytest.approx(value)


@pytest.mark.numba
def test_numba_obj_invalid_combinations():
    numba = pytest.importorskip("numba")
    pytest.importorskip("vector.backends._numba_object")

    @numba.njit
    def duplicate():
        return vector.obj(x=1.0, px=2.0, y=3.0)

    @numba.njit
    def two_temporal():
        return vector.obj(pt=1.0, phi=2.0, eta=3.0, mass=4.0, energy=5.0)

    @numba.njit
    def two_longitudinal():
        return vector.obj(x=1.0, y=2.0, theta=3.0, eta=4.0)

    @numba.njit
    def two_azimuthal():
        return vector.obj(x=1.0, y=2.0, rho=3.0, phi=4.0)

    for function, complaint in (
        (duplicate, "duplicate coordinates"),
        (two_temporal, "specify t= or tau="),
        (two_longitudinal, "specify z= or theta= or eta="),
        (two_azimuthal, "specify x= and y= or rho= and phi="),
    ):
        with pytest.raises(numba.TypingError, match=complaint):
            function()
