# Copyright (c) 2019-2024, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
Defines behaviors for Awkward Array. New arrays created with the

.. code-block:: python

    vector.Array(...)
    vector.awk(...)
    vector.zip(...)

function will have these behaviors built in (and will pass them to any derived
arrays).

Alternatively, you can

.. code-block:: python

   vector.register_awkward()

to install the behaviors globally, so that any record named ``Vector2D``,
``Vector3D``, ``Vector4D``, ``Momentum2D``, ``Momentum3D``, or ``Momentum4D``
will have these properties and methods.

The Awkward-Vectors-in-Numba extension is also implemented here, since it requires
two non-strict dependencies of Vector: Awkward and Numba. Awkward's ``ak.behavior``
manages this non-strictness well.
"""

from __future__ import annotations

import numbers
import types
import typing

import awkward as ak
import numpy

import vector
from vector._methods import (
    Azimuthal,
    AzimuthalRhoPhi,
    AzimuthalXY,
    Longitudinal,
    LongitudinalEta,
    LongitudinalTheta,
    LongitudinalZ,
    Lorentz,
    LorentzMomentum,
    Momentum,
    Planar,
    PlanarMomentum,
    Spatial,
    SpatialMomentum,
    Temporal,
    TemporalT,
    TemporalTau,
    Vector2D,
    Vector3D,
    Vector4D,
    VectorProtocol,
)
from vector._typeutils import BoolCollection, Protocol, ScalarCollection
from vector.backends.numpy import VectorNumpy2D, VectorNumpy3D, VectorNumpy4D
from vector.backends.object import (
    AzimuthalObjectRhoPhi,
    AzimuthalObjectXY,
    LongitudinalObjectEta,
    LongitudinalObjectTheta,
    LongitudinalObjectZ,
    TemporalObjectT,
    TemporalObjectTau,
    VectorObject2D,
    VectorObject3D,
    VectorObject4D,
)

# Throws an error if awkward is too old
vector._import_awkward()

ArrayOrRecord = typing.TypeVar("ArrayOrRecord", bound=typing.Union[ak.Array, ak.Record])

behavior: typing.Any = {}


# coordinates classes are a formality for Awkward #############################


class CoordinatesAwkward:
    lib: types.ModuleType = numpy


class AzimuthalAwkward(CoordinatesAwkward, Azimuthal):
    """
    Azimuthal class for the Awkward backend.
    See -

    - :meth:`AzimuthalAwkward.from_fields`
    - :meth:`AzimuthalAwkward.from_momentum_fields`

    to construct azimuthal type objects.
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self.elements}"

    @classmethod
    def from_fields(cls, array: ak.Array) -> AzimuthalAwkward:
        """
        Create a :class:`vector.backends.awkward.AzimuthalAwkwardXY` or a
        :class:`vector.backends.awkward.AzimuthalAwkwardRhoPhi`, depending on
        the fields in ``array``.

        Examples:
            >>> import vector
            >>> import awkward as ak
            >>> a = ak.Array([{"x": [1, 2]}, {"y": [1]}])
            >>> az = vector.backends.awkward.AzimuthalAwkward.from_fields(a)
            >>> az
            AzimuthalAwkwardXY(<Array [[1, 2], None] type='2 * option[var * int64]'>, <Array [None, [1]] type='2 * option[var * int64]'>)
            >>> az.elements
            (<Array [[1, 2], None] type='2 * option[var * int64]'>, <Array [None, [1]] type='2 * option[var * int64]'>)
        """
        fields = ak.fields(array)
        if "x" in fields and "y" in fields:
            return AzimuthalAwkwardXY(array["x"], array["y"])
        elif "rho" in fields and "phi" in fields:
            return AzimuthalAwkwardRhoPhi(array["rho"], array["phi"])
        else:
            raise ValueError(
                "array does not have azimuthal coordinates (x, y or rho, phi): "
                f"{', '.join(fields)}"
            )

    @classmethod
    def from_momentum_fields(cls, array: ak.Array) -> AzimuthalAwkward:
        """
        Create a :class:`vector.backends.awkward.AzimuthalAwkwardXY` or a
        :class:`vector.backends.awkward.AzimuthalAwkwardRhoPhi`, depending on
        the fields in ``array``, allowing momentum synonyms.

        Examples:
            >>> import vector
            >>> import awkward as ak
            >>> a = ak.Array([{"px": [1, 2]}, {"py": [1]}])
            >>> az = vector.backends.awkward.AzimuthalAwkward.from_momentum_fields(a)
            >>> az
            AzimuthalAwkwardXY(<Array [[1, 2], None] type='2 * option[var * int64]'>, <Array [None, [1]] type='2 * option[var * int64]'>)
            >>> az.elements
            (<Array [[1, 2], None] type='2 * option[var * int64]'>, <Array [None, [1]] type='2 * option[var * int64]'>)
        """
        fields = ak.fields(array)
        if "x" in fields and "y" in fields:
            return AzimuthalAwkwardXY(array["x"], array["y"])
        elif "x" in fields and "py" in fields:
            return AzimuthalAwkwardXY(array["x"], array["py"])
        elif "px" in fields and "y" in fields:
            return AzimuthalAwkwardXY(array["px"], array["y"])
        elif "px" in fields and "py" in fields:
            return AzimuthalAwkwardXY(array["px"], array["py"])
        elif "rho" in fields and "phi" in fields:
            return AzimuthalAwkwardRhoPhi(array["rho"], array["phi"])
        elif "pt" in fields and "phi" in fields:
            return AzimuthalAwkwardRhoPhi(array["pt"], array["phi"])
        else:
            raise ValueError(
                "array does not have azimuthal coordinates (x/px, y/py or rho/pt, phi): "
                f"{', '.join(fields)}"
            )


class LongitudinalAwkward(CoordinatesAwkward, Longitudinal):
    """
    Longitudinal class for the Awkward backend.
    See -

    - :meth:`LongitudinalAwkward.from_fields`
    - :meth:`LongitudinalAwkward.from_momentum_fields`

    to construct longitudinal type objects.
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self.elements}"

    @classmethod
    def from_fields(cls, array: ak.Array) -> LongitudinalAwkward:
        """
        Create a :class:`vector.backends.awkward.LongitudinalAwkwardZ`, a
        :class:`vector.backends.awkward.LongitudinalAwkwardTheta`, or a
        :class:`vector.backends.awkward.LongitudinalAwkwardEta`, depending on
        the fields in ``array``.

        Examples:
            >>> import vector
            >>> import awkward as ak
            >>> a = ak.Array([{"theta": [1, 0]}])
            >>> l = vector.backends.awkward.LongitudinalAwkward.from_fields(a)
            >>> l
            LongitudinalAwkwardTheta(<Array [[1, 0]] type='1 * var * int64'>,)
            >>> l.elements
            (<Array [[1, 0]] type='1 * var * int64'>,)
        """
        fields = ak.fields(array)
        if "z" in fields:
            return LongitudinalAwkwardZ(array["z"])
        elif "theta" in fields:
            return LongitudinalAwkwardTheta(array["theta"])
        elif "eta" in fields:
            return LongitudinalAwkwardEta(array["eta"])
        else:
            raise ValueError(
                "array does not have longitudinal coordinates (z or theta or eta): "
                f"{', '.join(fields)}"
            )

    @classmethod
    def from_momentum_fields(cls, array: ak.Array) -> LongitudinalAwkward:
        """
        Create a :class:`vector.backends.awkward.LongitudinalAwkwardZ`, a
        :class:`vector.backends.awkward.LongitudinalAwkwardTheta`, or a
        :class:`vector.backends.awkward.LongitudinalAwkwardEta`, depending on
        the fields in ``array``, allowing momentum synonyms.

        Examples:
            >>> import vector
            >>> import awkward as ak
            >>> a = ak.Array([{"theta": [1, 0]}])
            >>> l = vector.backends.awkward.LongitudinalAwkward.from_momentum_fields(a)
            >>> l
            LongitudinalAwkwardTheta(<Array [[1, 0]] type='1 * var * int64'>,)
            >>> l.elements
            (<Array [[1, 0]] type='1 * var * int64'>,)
        """
        fields = ak.fields(array)
        if "z" in fields:
            return LongitudinalAwkwardZ(array["z"])
        elif "pz" in fields:
            return LongitudinalAwkwardZ(array["pz"])
        elif "theta" in fields:
            return LongitudinalAwkwardTheta(array["theta"])
        elif "eta" in fields:
            return LongitudinalAwkwardEta(array["eta"])
        else:
            raise ValueError(
                "array does not have longitudinal coordinates (z/pz or theta or eta): "
                f"{', '.join(fields)}"
            )


class TemporalAwkward(CoordinatesAwkward, Temporal):
    """
    Temporal class for the Awkward backend.
    See -

    - :meth:`TemporalAwkward.from_fields`
    - :meth:`TemporalAwkward.from_momentum_fields`

    to construct longitudinal type objects.
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self.elements}"

    @classmethod
    def from_fields(cls, array: ak.Array) -> TemporalAwkward:
        """
        Create a :class:`vector.backends.awkward.TemporalT` or a
        :class:`vector.backends.awkward.TemporalTau`, depending on
        the fields in ``array``.

        Examples:
            >>> import vector
            >>> import awkward as ak
            >>> a = ak.Array([{"tau": [1, 0]}])
            >>> t = vector.backends.awkward.TemporalAwkward.from_fields(a)
            >>> t
            TemporalAwkwardTau(<Array [[1, 0]] type='1 * var * int64'>,)
            >>> t.elements
            (<Array [[1, 0]] type='1 * var * int64'>,)
        """
        fields = ak.fields(array)
        if "t" in fields:
            return TemporalAwkwardT(array["t"])
        elif "tau" in fields:
            return TemporalAwkwardTau(array["tau"])
        else:
            raise ValueError(
                "array does not have temporal coordinates (t or tau): "
                f"{', '.join(fields)}"
            )

    @classmethod
    def from_momentum_fields(cls, array: ak.Array) -> TemporalAwkward:
        """
        Create a :class:`vector.backends.awkward.TemporalT` or a
        :class:`vector.backends.awkward.TemporalTau`, depending on
        the fields in ``array``, allowing momentum synonyms.

        Examples:
            >>> import vector
            >>> import awkward as ak
            >>> a = ak.Array([{"mass": [1, 0]}])
            >>> t = vector.backends.awkward.TemporalAwkward.from_momentum_fields(a)
            >>> t
            TemporalAwkwardTau(<Array [[1, 0]] type='1 * var * int64'>,)
            >>> t.elements
            (<Array [[1, 0]] type='1 * var * int64'>,)
        """
        fields = ak.fields(array)
        if "t" in fields:
            return TemporalAwkwardT(array["t"])
        elif "E" in fields:
            return TemporalAwkwardT(array["E"])
        elif "e" in fields:
            return TemporalAwkwardT(array["e"])
        elif "energy" in fields:
            return TemporalAwkwardT(array["energy"])
        elif "tau" in fields:
            return TemporalAwkwardTau(array["tau"])
        elif "M" in fields:
            return TemporalAwkwardTau(array["M"])
        elif "m" in fields:
            return TemporalAwkwardTau(array["m"])
        elif "mass" in fields:
            return TemporalAwkwardTau(array["mass"])
        else:
            raise ValueError(
                "array does not have temporal coordinates (t/E/e/energy or tau/M/m/mass): "
                f"{', '.join(fields)}"
            )


class AzimuthalAwkwardXY(AzimuthalAwkward, AzimuthalXY):
    """
    Class for the ``x`` and ``y`` (azimuthal) coordinates of Awkward backend.

    Examples:
        >>> import vector
        >>> import awkward as ak
        >>> a = ak.Array([{"x": [1, 2]}, {"y": [1]}])
        >>> az = vector.backends.awkward.AzimuthalAwkwardXY(a["x"], a["y"])
        >>> az
        AzimuthalAwkwardXY(<Array [[1, 2], None] type='2 * option[var * int64]'>, <Array [None, [1]] type='2 * option[var * int64]'>)
        >>> az.elements
        (<Array [[1, 2], None] type='2 * option[var * int64]'>, <Array [None, [1]] type='2 * option[var * int64]'>)
    """

    __slots__ = ("x", "y")

    def __init__(self, x: typing.Any, y: typing.Any) -> None:
        self.x = x
        self.y = y

    @property
    def elements(self) -> tuple[ArrayOrRecord, ArrayOrRecord]:
        """
        Azimuthal coordinates (``x`` and ``y``) as a tuple.

        Examples:
            >>> import vector
            >>> az = vector.backends.awkward.AzimuthalAwkwardXY([1, 2, 3], [1, 2])
            >>> az.elements
            ([1, 2, 3], [1, 2])
        """
        return (self.x, self.y)


class AzimuthalAwkwardRhoPhi(AzimuthalAwkward, AzimuthalRhoPhi):
    """
    Class for the ``rho`` and ``phi`` (azimuthal) coordinates of Awkward backend.

    Examples:
        >>> import vector
        >>> import awkward as ak
        >>> a = ak.Array([{"rho": [1, 2]}, {"phi": [1]}])
        >>> az = vector.backends.awkward.AzimuthalAwkwardRhoPhi(a["rho"], a["phi"])
        >>> az
        AzimuthalAwkwardRhoPhi(<Array [[1, 2], None] type='2 * option[var * int64]'>, <Array [None, [1]] type='2 * option[var * int64]'>)
        >>> az.elements
        (<Array [[1, 2], None] type='2 * option[var * int64]'>, <Array [None, [1]] type='2 * option[var * int64]'>)
    """

    __slots__ = ("rho", "phi")

    def __init__(self, rho: typing.Any, phi: typing.Any) -> None:
        self.rho = rho
        self.phi = phi

    @property
    def elements(self) -> tuple[ArrayOrRecord, ArrayOrRecord]:
        """
        Azimuthal coordinates (``rho`` and ``phi``) as a tuple.

        Examples:
            >>> import vector
            >>> az = vector.backends.awkward.AzimuthalAwkwardRhoPhi([1, 2, 3], [1, 2])
            >>> az.elements
            ([1, 2, 3], [1, 2])
        """
        return (self.rho, self.phi)


class LongitudinalAwkwardZ(LongitudinalAwkward, LongitudinalZ):
    """
    Class for the ``z`` (longitudinal) coordinate of Awkward backend.

    Examples:
        >>> import vector
        >>> import awkward as ak
        >>> a = ak.Array([{"z": [1, 2]}])
        >>> l = vector.backends.awkward.LongitudinalAwkwardZ(a["z"])
        >>> l
        LongitudinalAwkwardZ(<Array [[1, 2]] type='1 * var * int64'>,)
        >>> l.elements
        (<Array [[1, 2]] type='1 * var * int64'>,)
    """

    __slots__ = ("z",)

    def __init__(self, z: typing.Any) -> None:
        self.z = z

    @property
    def elements(self) -> tuple[ArrayOrRecord]:
        """
        Longitudinal coordinates (``z``) as a tuple.

        Examples:
            >>> import vector
            >>> l = vector.backends.awkward.LongitudinalAwkwardZ(5)
            >>> l.elements
            (5,)
        """
        return (self.z,)


class LongitudinalAwkwardTheta(LongitudinalAwkward, LongitudinalTheta):
    """
    Class for the ``theta`` (longitudinal) coordinate of Awkward backend.

    Examples:
        >>> import vector
        >>> import awkward as ak
        >>> a = ak.Array([{"theta": [1, 2]}])
        >>> l = vector.backends.awkward.LongitudinalAwkwardTheta(a["theta"])
        >>> l
        LongitudinalAwkwardTheta(<Array [[1, 2]] type='1 * var * int64'>,)
        >>> l.elements
        (<Array [[1, 2]] type='1 * var * int64'>,)
    """

    __slots__ = ("theta",)

    def __init__(self, theta: typing.Any) -> None:
        self.theta = theta

    @property
    def elements(self) -> tuple[ArrayOrRecord]:
        """
        Longitudinal coordinates (``theta``) as a tuple.

        Examples:
            >>> import vector
            >>> l = vector.backends.awkward.LongitudinalAwkwardTheta(5)
            >>> l.elements
            (5,)
        """
        return (self.theta,)


class LongitudinalAwkwardEta(LongitudinalAwkward, LongitudinalEta):
    """
    Class for the ``eta`` (longitudinal) coordinate of Awkward backend.

    Examples:
        >>> import vector
        >>> import awkward as ak
        >>> a = ak.Array([{"eta": [1, 2]}])
        >>> l = vector.backends.awkward.LongitudinalAwkwardEta(a["eta"])
        >>> l
        LongitudinalAwkwardEta(<Array [[1, 2]] type='1 * var * int64'>,)
        >>> l.elements
        (<Array [[1, 2]] type='1 * var * int64'>,)
    """

    __slots__ = ("eta",)

    def __init__(self, eta: typing.Any) -> None:
        self.eta = eta

    @property
    def elements(self) -> tuple[ArrayOrRecord]:
        """
        Longitudinal coordinates (``eta``) as a tuple.

        Examples:
            >>> import vector
            >>> l = vector.backends.awkward.LongitudinalAwkwardEta(5)
            >>> l.elements
            (5,)
        """
        return (self.eta,)


class TemporalAwkwardT(TemporalAwkward, TemporalT):
    """
    Class for the ``t`` (temporal) coordinate of Awkward backend.

    Examples:
        >>> import vector
        >>> import awkward as ak
        >>> a = ak.Array([{"t": [1, 2]}])
        >>> t = vector.backends.awkward.TemporalAwkwardT(a["t"])
        >>> t
        TemporalAwkwardT(<Array [[1, 2]] type='1 * var * int64'>,)
        >>> t.elements
        (<Array [[1, 2]] type='1 * var * int64'>,)
    """

    __slots__ = ("t",)

    def __init__(self, t: typing.Any) -> None:
        self.t = t

    @property
    def elements(self) -> tuple[ArrayOrRecord]:
        """
        Temporal coordinates (``t``) as a tuple.

        Examples:
            >>> import vector
            >>> t = vector.backends.awkward.TemporalAwkwardT(5)
            >>> t.elements
            (5,)
        """
        return (self.t,)


class TemporalAwkwardTau(TemporalAwkward, TemporalTau):
    """
    Class for the ``tau`` (temporal) coordinate of Awkward backend.

    Examples:
        >>> import vector
        >>> import awkward as ak
        >>> a = ak.Array([{"tau": [1, 2]}])
        >>> t = vector.backends.awkward.TemporalAwkwardTau(a["tau"])
        >>> t
        TemporalAwkwardTau(<Array [[1, 2]] type='1 * var * int64'>,)
        >>> t.elements
        (<Array [[1, 2]] type='1 * var * int64'>,)
    """

    __slots__ = ("tau",)

    def __init__(self, tau: typing.Any) -> None:
        self.tau = tau

    @property
    def elements(self) -> tuple[ArrayOrRecord]:
        """
        Temporal coordinates (``tau``) as a tuple.

        Examples:
            >>> import vector
            >>> t = vector.backends.awkward.TemporalAwkwardTau(5)
            >>> t.elements
            (5,)
        """
        return (self.tau,)


def _class_to_name(cls: type[VectorProtocol]) -> str:
    # respect the type of classes inheriting VectorAwkward classes
    is_vector = "vector.backends" in cls.__module__
    if issubclass(cls, Momentum):
        if issubclass(cls, Vector2D):
            return "Momentum2D" if is_vector else cls.__name__[:-5]
        if issubclass(cls, Vector3D):
            return "Momentum3D" if is_vector else cls.__name__[:-5]
        if issubclass(cls, Vector4D):
            return "Momentum4D" if is_vector else cls.__name__[:-5]
    if issubclass(cls, Vector2D):
        return "Vector2D" if is_vector else cls.__name__[:-5]
    if issubclass(cls, Vector3D):
        return "Vector3D" if is_vector else cls.__name__[:-5]
    if issubclass(cls, Vector4D):
        return "Vector4D" if is_vector else cls.__name__[:-5]

    raise AssertionError(repr(cls))


# the vector class ############################################################


def _yes_record(
    x: ak.Array,
) -> float | ak.Record | None:
    return x[0]


def _no_record(x: ak.Array) -> ak.Array | None:
    return x


# Type for mixing in Awkward later
class AwkwardProtocol(Protocol):
    def __getitem__(self, where: typing.Any) -> float | ak.Array | ak.Record | None: ...


class VectorAwkward:
    """Mixin class for Awkward vectors."""

    lib: types.ModuleType = numpy

    def _wrap_result(
        self: AwkwardProtocol,
        cls: typing.Any,
        result: typing.Any,
        returns: typing.Any,
        num_vecargs: typing.Any,
    ) -> typing.Any:
        """
        Wraps the raw result of a compute function as an array of scalars or an
        array of vectors.

        Args:
            result: Value or tuple of values from a compute function.
            returns: Signature from a ``dispatch_map``.
            num_vecargs (int): Number of vector arguments in the function
                that would be treated on an equal footing (i.e. ``add``
                has two, but ``rotate_axis`` has only one: the ``axis``
                is secondary).
        """
        if returns in ([float], [bool]):
            return result

        if all(not isinstance(x, ak.Array) for x in result):
            maybe_record = _yes_record
            result = [
                ak.Array(x.layout.array[x.layout.at : x.layout.at + 1])
                if isinstance(x, ak.Record)
                else ak.Array([x])
                for x in result
            ]
        else:
            maybe_record = _no_record

        if (
            len(returns) == 1
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
        ):
            first = next(x for x in result if isinstance(x, ak.Array))
            result = [
                x if isinstance(x, ak.Array) else ak.broadcast_arrays(first, x)[1]
                for x in result
            ]

            names = []
            arrays = []
            if returns[0] is AzimuthalXY:
                names.extend(["x", "y"])
                arrays.extend([result[0], result[1]])
            elif returns[0] is AzimuthalRhoPhi:
                names.extend(["rho", "phi"])
                arrays.extend([result[0], result[1]])

            fields = ak.fields(self)
            if num_vecargs == 1:
                for name in fields:
                    if name not in (
                        "x",
                        "y",
                        "rho",
                        "pt",
                        "phi",
                    ):
                        names.append(name)
                        arrays.append(self[name])

            if "t" in fields or "tau" in fields:
                cls = cls.ProjectionClass4D
            elif "z" in fields or "theta" in fields or "eta" in fields:
                cls = cls.ProjectionClass3D
            else:
                cls = cls.ProjectionClass2D

            return maybe_record(
                ak.zip(
                    dict(zip(names, arrays)),
                    depth_limit=first.layout.purelist_depth,
                    with_name=_class_to_name(cls),
                    behavior=None if vector._awkward_registered else first.behavior,
                )
            )

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and returns[1] is None
        ):
            first = next(x for x in result if isinstance(x, ak.Array))
            result = [
                x if isinstance(x, ak.Array) else ak.broadcast_arrays(first, x)[1]
                for x in result
            ]

            names = []
            arrays = []
            if returns[0] is AzimuthalXY:
                names.extend(["x", "y"])
                arrays.extend([result[0], result[1]])
            elif returns[0] is AzimuthalRhoPhi:
                names.extend(["rho", "phi"])
                arrays.extend([result[0], result[1]])

            if num_vecargs == 1:
                for name in ak.fields(self):
                    if name not in (
                        "x",
                        "y",
                        "rho",
                        "pt",
                        "phi",
                        "z",
                        "pz",
                        "theta",
                        "eta",
                        "t",
                        "tau",
                        "m",
                        "M",
                        "mass",
                        "e",
                        "E",
                        "energy",
                    ):
                        names.append(name)
                        arrays.append(self[name])

            return maybe_record(
                ak.zip(
                    dict(zip(names, arrays)),
                    depth_limit=first.layout.purelist_depth,
                    with_name=_class_to_name(cls.ProjectionClass2D),
                    behavior=None if vector._awkward_registered else first.behavior,
                )
            )

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
        ):
            first = next(x for x in result if isinstance(x, ak.Array))
            result = [
                x if isinstance(x, ak.Array) else ak.broadcast_arrays(first, x)[1]
                for x in result
            ]

            names = []
            arrays = []
            if returns[0] is AzimuthalXY:
                names.extend(["x", "y"])
                arrays.extend([result[0], result[1]])
            elif returns[0] is AzimuthalRhoPhi:
                names.extend(["rho", "phi"])
                arrays.extend([result[0], result[1]])

            if returns[1] is LongitudinalZ:
                names.append("z")
                arrays.append(result[2])
            elif returns[1] is LongitudinalTheta:
                names.append("theta")
                arrays.append(result[2])
            elif returns[1] is LongitudinalEta:
                names.append("eta")
                arrays.append(result[2])

            fields = ak.fields(self)
            if num_vecargs == 1:
                for name in fields:
                    if name not in (
                        "x",
                        "y",
                        "rho",
                        "pt",
                        "phi",
                        "z",
                        "pz",
                        "theta",
                        "eta",
                    ):
                        names.append(name)
                        arrays.append(self[name])

            if "t" in fields or "tau" in fields:
                cls = cls.ProjectionClass4D
            else:
                cls = cls.ProjectionClass3D

            return maybe_record(
                ak.zip(
                    dict(zip(names, arrays)),
                    depth_limit=first.layout.purelist_depth,
                    with_name=_class_to_name(cls),
                    behavior=None if vector._awkward_registered else first.behavior,
                )
            )

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and returns[2] is None
        ):
            first = next(x for x in result if isinstance(x, ak.Array))
            result = [
                x if isinstance(x, ak.Array) else ak.broadcast_arrays(first, x)[1]
                for x in result
            ]

            names = []
            arrays = []
            if returns[0] is AzimuthalXY:
                names.extend(["x", "y"])
                arrays.extend([result[0], result[1]])
            elif returns[0] is AzimuthalRhoPhi:
                names.extend(["rho", "phi"])
                arrays.extend([result[0], result[1]])

            if returns[1] is LongitudinalZ:
                names.append("z")
                arrays.append(result[2])
            elif returns[1] is LongitudinalTheta:
                names.append("theta")
                arrays.append(result[2])
            elif returns[1] is LongitudinalEta:
                names.append("eta")
                arrays.append(result[2])

            if num_vecargs == 1:
                for name in ak.fields(self):
                    if name not in (
                        "x",
                        "y",
                        "rho",
                        "pt",
                        "phi",
                        "z",
                        "pz",
                        "theta",
                        "eta",
                        "t",
                        "tau",
                        "m",
                        "M",
                        "mass",
                        "e",
                        "E",
                        "energy",
                    ):
                        names.append(name)
                        arrays.append(self[name])

            return maybe_record(
                ak.zip(
                    dict(zip(names, arrays)),
                    depth_limit=first.layout.purelist_depth,
                    with_name=_class_to_name(cls.ProjectionClass3D),
                    behavior=None if vector._awkward_registered else first.behavior,
                )
            )

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], Longitudinal)
            and isinstance(returns[2], type)
            and issubclass(returns[2], Temporal)
        ):
            first = next(x for x in result if isinstance(x, ak.Array))
            result = [
                x if isinstance(x, ak.Array) else ak.broadcast_arrays(first, x)[1]
                for x in result
            ]

            names = []
            arrays = []
            if returns[0] is AzimuthalXY:
                names.extend(["x", "y"])
                arrays.extend([result[0], result[1]])
            elif returns[0] is AzimuthalRhoPhi:
                names.extend(["rho", "phi"])
                arrays.extend([result[0], result[1]])

            if returns[1] is LongitudinalZ:
                names.append("z")
                arrays.append(result[2])
            elif returns[1] is LongitudinalTheta:
                names.append("theta")
                arrays.append(result[2])
            elif returns[1] is LongitudinalEta:
                names.append("eta")
                arrays.append(result[2])

            if returns[2] is TemporalT:
                names.append("t")
                arrays.append(result[3])
            elif returns[2] is TemporalTau:
                names.append("tau")
                arrays.append(result[3])

            if num_vecargs == 1:
                for name in ak.fields(self):
                    if name not in (
                        "x",
                        "y",
                        "rho",
                        "pt",
                        "phi",
                        "z",
                        "pz",
                        "theta",
                        "eta",
                        "t",
                        "tau",
                        "m",
                        "M",
                        "mass",
                        "e",
                        "E",
                        "energy",
                    ):
                        names.append(name)
                        arrays.append(self[name])

            return maybe_record(
                ak.zip(
                    dict(zip(names, arrays)),
                    depth_limit=first.layout.purelist_depth,
                    with_name=_class_to_name(cls.ProjectionClass4D),
                    behavior=None if vector._awkward_registered else first.behavior,
                )
            )

        else:
            raise AssertionError(repr(returns))


class VectorAwkward2D(VectorAwkward, Planar, Vector2D):
    """
    Two dimensional vector class for the Awkward backend.
    Two dimensional awkward vectors for the users are defined using the
    :class:`VectorArray2D` class.

    See :class:`MomentumAwkward2D` for momentum vectors.
    """

    @property
    def azimuthal(self) -> AzimuthalAwkward:
        """
        Returns an Azimuthal type object.

        Use the ``elements`` property of the created object
        to access the coordinates.

        Examples:
            >>> import vector
            >>> a = vector.Array(
            ...      [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
            ... )
            >>> a.azimuthal
            AzimuthalAwkwardXY(<Array [1, 2] type='2 * int64'>, <Array [1.1, 2.2] type='2 * float64'>)
            >>> a.azimuthal.elements
            (<Array [1, 2] type='2 * int64'>, <Array [1.1, 2.2] type='2 * float64'>)
        """
        return AzimuthalAwkward.from_fields(self)


class MomentumAwkward2D(PlanarMomentum, VectorAwkward2D):
    """
    Two dimensional momentum vector class for the Awkward backend.
    Two dimensional momentum vectors for the users are defined using the
    :class:`MomentumArray2D` class.

    See :class:`VectorAwkward2D` for momentum vectors.
    """

    @property
    def azimuthal(self) -> AzimuthalAwkward:
        """
        Returns an Azimuthal type object containing the azimuthal coordinates.

        Use the ``elements`` property of the created object
        to access the coordinates.

        Examples:
            >>> import vector
            >>> a = vector.Array(
            ...      [{"px": 1, "py": 1.1}, {"px": 2, "py": 2.2}],
            ... )
            >>> a.azimuthal
            AzimuthalAwkwardXY(<Array [1, 2] type='2 * int64'>, <Array [1.1, 2.2] type='2 * float64'>)
            >>> a.azimuthal.elements
            (<Array [1, 2] type='2 * int64'>, <Array [1.1, 2.2] type='2 * float64'>)
        """
        return AzimuthalAwkward.from_momentum_fields(self)


class VectorAwkward3D(VectorAwkward, Spatial, Vector3D):
    """
    Three dimensional vector class for the Awkward backend.
    Three dimensional awkward vectors for the users are defined using the
    :class:`VectorArray3D` class.

    See :class:`MomentumAwkward3D` for momentum vectors.
    """

    @property
    def azimuthal(self) -> AzimuthalAwkward:
        """
        Returns an Azimuthal type object containing the azimuthal coordinates.

        Use the ``elements`` property of the created object
        to access the coordinates.

        Examples:
            >>> import vector
            >>> a = vector.Array(
            ...       [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": 0.2}],
            ... )
            >>> a.azimuthal
            AzimuthalAwkwardXY(<Array [1, 2] type='2 * int64'>, <Array [1.1, 2.2] type='2 * float64'>)
            >>> a.azimuthal.elements
            (<Array [1, 2] type='2 * int64'>, <Array [1.1, 2.2] type='2 * float64'>)
        """
        return AzimuthalAwkward.from_fields(self)

    @property
    def longitudinal(self) -> LongitudinalAwkward:
        """
        Returns a Longitudinal type object containing the longitudinal coordinates.

        Use the ``elements`` property of the created object
        to access the coordinates.

        Examples:
            >>> import vector
            >>> a = vector.Array(
            ...       [{"x": 1, "y": 1.1, "z": 0.1}, {"x": 2, "y": 2.2, "z": 0.2}],
            ... )
            >>> a.longitudinal
            LongitudinalAwkwardZ(<Array [0.1, 0.2] type='2 * float64'>,)
            >>> a.longitudinal.elements
            (<Array [0.1, 0.2] type='2 * float64'>,)
        """
        return LongitudinalAwkward.from_fields(self)


class MomentumAwkward3D(SpatialMomentum, VectorAwkward3D):
    """
    Three dimensional momentum vector class for the Awkward backend.
    Three dimensional momentum vectors for the users are defined using the
    :class:`MomentumArray3D` class.

    See :class:`VectorAwkward3D` for momentum vectors.
    """

    @property
    def azimuthal(self) -> AzimuthalAwkward:
        """
        Returns an Azimuthal type object containing the azimuthal coordinates.

        Use the ``elements`` property of the created object
        to access the coordinates.

        Examples:
            >>> import vector
            >>> a = vector.Array(
            ...       [{"px": 1, "py": 1.1, "z": 0.1}, {"px": 2, "py": 2.2, "z": 0.2}],
            ... )
            >>> a.azimuthal
            AzimuthalAwkwardXY(<Array [1, 2] type='2 * int64'>, <Array [1.1, 2.2] type='2 * float64'>)
            >>> a.azimuthal.elements
            (<Array [1, 2] type='2 * int64'>, <Array [1.1, 2.2] type='2 * float64'>)
        """
        return AzimuthalAwkward.from_momentum_fields(self)

    @property
    def longitudinal(self) -> LongitudinalAwkward:
        """
        Returns a Longitudinal type object containing the longitudinal coordinates.

        Use the ``elements`` property of the created object
        to access the coordinates.

        Examples:
            >>> import vector
            >>> a = vector.Array(
            ...       [{"px": 1, "py": 1.1, "z": 0.1}, {"px": 2, "py": 2.2, "z": 0.2}],
            ... )
            >>> a.longitudinal
            LongitudinalAwkwardZ(<Array [0.1, 0.2] type='2 * float64'>,)
            >>> a.longitudinal.elements
            (<Array [0.1, 0.2] type='2 * float64'>,)
        """
        return LongitudinalAwkward.from_momentum_fields(self)


class VectorAwkward4D(VectorAwkward, Lorentz, Vector4D):
    """
    Four dimensional momentum vector class for the Awkward backend.
    Four dimensional momentum vectors for the users are defined using the
    :class:`MomentumArray4D` class.

    See :class:`VectorAwkward4D` for momentum vectors.
    """

    @property
    def azimuthal(self) -> AzimuthalAwkward:
        """
        Returns an Azimuthal type object containing the azimuthal coordinates.

        Use the ``elements`` property of the created object
        to access the coordinates.

        Examples:
            >>> import vector
            >>> a = vector.Array(
            ...       [{"x": 1, "y": 1.1, "z": 0.1, "tau": 1}, {"x": 2, "y": 2.2, "z": 0.2, "tau": 3}],
            ... )
            >>> a.azimuthal
            AzimuthalAwkwardXY(<Array [1, 2] type='2 * int64'>, <Array [1.1, 2.2] type='2 * float64'>)
            >>> a.azimuthal.elements
            (<Array [1, 2] type='2 * int64'>, <Array [1.1, 2.2] type='2 * float64'>)
        """
        return AzimuthalAwkward.from_fields(self)

    @property
    def longitudinal(self) -> LongitudinalAwkward:
        """
        Returns a Longitudinal type object containing the longitudinal coordinates.

        Use the ``elements`` property of the created object
        to access the coordinates.

        Examples:
            >>> import vector
            >>> a = vector.Array(
            ...       [{"x": 1, "y": 1.1, "z": 0.1, "tau": 1}, {"x": 2, "y": 2.2, "z": 0.2, "tau": 3}],
            ... )
            >>> a.longitudinal
            LongitudinalAwkwardZ(<Array [0.1, 0.2] type='2 * float64'>,)
            >>> a.longitudinal.elements
            (<Array [0.1, 0.2] type='2 * float64'>,)
        """
        return LongitudinalAwkward.from_fields(self)

    @property
    def temporal(self) -> TemporalAwkward:
        """
        Returns a Temporal type object containing the temporal coordinates.

        Use the ``elements`` property of the created object
        to access the coordinates.

        Examples:
            >>> import vector
            >>> a = vector.Array(
            ...       [{"x": 1, "y": 1.1, "z": 0.1, "tau": 1}, {"x": 2, "y": 2.2, "z": 0.2, "tau": 3}],
            ... )
            >>> a.temporal
            TemporalAwkwardTau(<Array [1, 3] type='2 * int64'>,)
            >>> a.temporal.elements
            (<Array [1, 3] type='2 * int64'>,)
        """
        return TemporalAwkward.from_fields(self)


class MomentumAwkward4D(LorentzMomentum, VectorAwkward4D):
    """
    Four dimensional momentum vector class for the Awkward backend.
    Four dimensional momentum vectors for the users are defined using the
    :class:`MomentumArray4D` class.

    See :class:`VectorAwkward4D` for momentum vectors.
    """

    @property
    def azimuthal(self) -> AzimuthalAwkward:
        """
        Returns an Azimuthal type object containing the azimuthal coordinates.

        Use the ``elements`` property of the created object
        to access the coordinates.

        Examples:
            >>> import vector
            >>> a = vector.Array(
            ...       [{"px": 1, "py": 1.1, "z": 0.1, "m": 1}, {"px": 2, "py": 2.2, "z": 0.2, "m": 3}],
            ... )
            >>> a.azimuthal
            AzimuthalAwkwardXY(<Array [1, 2] type='2 * int64'>, <Array [1.1, 2.2] type='2 * float64'>)
            >>> a.azimuthal.elements
            (<Array [1, 2] type='2 * int64'>, <Array [1.1, 2.2] type='2 * float64'>)
        """
        return AzimuthalAwkward.from_momentum_fields(self)

    @property
    def longitudinal(self) -> LongitudinalAwkward:
        """
        Returns a Longitudinal type object containing the longitudinal coordinates.

        Use the ``elements`` property of the created object
        to access the coordinates.

        Examples:
            >>> import vector
            >>> a = vector.Array(
            ...       [{"px": 1, "py": 1.1, "z": 0.1, "m": 1}, {"px": 2, "py": 2.2, "z": 0.2, "m": 3}],
            ... )
            >>> a.longitudinal
            LongitudinalAwkwardZ(<Array [0.1, 0.2] type='2 * float64'>,)
            >>> a.longitudinal.elements
            (<Array [0.1, 0.2] type='2 * float64'>,)
        """
        return LongitudinalAwkward.from_momentum_fields(self)

    @property
    def temporal(self) -> TemporalAwkward:
        """
        Returns a Temporal type object containing the temporal coordinates.

        Use the ``elements`` property of the created object
        to access the coordinates.
        Examples:
            >>> import vector
            >>> a = vector.Array(
            ...       [{"px": 1, "py": 1.1, "z": 0.1, "m": 1}, {"px": 2, "py": 2.2, "z": 0.2, "m": 3}],
            ... )
            >>> a.temporal
            TemporalAwkwardTau(<Array [1, 3] type='2 * int64'>,)
            >>> a.temporal.elements
            (<Array [1, 3] type='2 * int64'>,)
        """
        return TemporalAwkward.from_momentum_fields(self)


# ak.Array and ak.Record subclasses ###########################################


class VectorArray2D(VectorAwkward2D, ak.Array):  # type: ignore[misc]
    """
    Defines ``awkward`` behavior for a 2 dimensional vector.

    See :class:`VectorRecord2D` and :class:`MomentumArray2D`
    for the corresponding ``Record`` and ``Momentum`` classes.
    """

    def allclose(
        self,
        other: VectorProtocol,
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        """Like ``np.ndarray.allclose``, but for VectorArray2D."""
        return ak.all(self.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan))


behavior["*", "Vector2D"] = VectorArray2D


class VectorRecord2D(VectorAwkward2D, ak.Record):  # type: ignore[misc]
    """
    Defines ``awkward`` behavior for a 2 dimensional vector record.

    See :class:`VectorArray2D` and :class:`MomentumArray2D`
    for the corresponding ``Vector`` and ``Momentum`` classes.
    """


behavior["Vector2D"] = VectorRecord2D


class VectorArray3D(VectorAwkward3D, ak.Array):  # type: ignore[misc]
    """
    Defines ``awkward`` behavior for a 3 dimensional vector.

    See :class:`VectorRecord3D` and :class:`MomentumArray3D`
    for the corresponding ``Record`` and ``Momentum`` classes.
    """

    def allclose(
        self,
        other: VectorProtocol,
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        """Like ``np.ndarray.allclose``, but for VectorArray3D."""
        return ak.all(self.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan))


behavior["*", "Vector3D"] = VectorArray3D


class VectorRecord3D(VectorAwkward3D, ak.Record):  # type: ignore[misc]
    """
    Defines ``awkward`` behavior for a 3 dimensional vector record.

    See :class:`VectorArray3D` and :class:`MomentumArray3D`
    for the corresponding ``Vector`` and ``Momentum`` classes.
    """


behavior["Vector3D"] = VectorRecord3D


class VectorArray4D(VectorAwkward4D, ak.Array):  # type: ignore[misc]
    """
    Defines ``awkward`` behavior for a 4 dimensional vector.

    See :class:`VectorRecord4D` and :class:`MomentumArray4D`
    for the corresponding ``Record`` and ``Momentum`` classes.
    """

    def allclose(
        self,
        other: VectorProtocol,
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        """Like ``np.ndarray.allclose``, but for VectorArray4D."""
        return ak.all(self.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan))


behavior["*", "Vector4D"] = VectorArray4D


class VectorRecord4D(VectorAwkward4D, ak.Record):  # type: ignore[misc]
    """
    Defines ``awkward`` behavior for a 4 dimensional vector record.

    See :class:`VectorArray4D` and :class:`MomentumArray4D`
    for the corresponding ``Vector`` and ``Momentum`` classes.
    """


behavior["Vector4D"] = VectorRecord4D


class MomentumArray2D(MomentumAwkward2D, ak.Array):  # type: ignore[misc]
    """
    Defines ``awkward`` behavior for a 2 dimensional momentum vector.

    See :class:`MomentumRecord2D` and :class:`VectorArray2D`
    for the corresponding ``Record`` and ``Vector`` classes.
    """

    def allclose(
        self,
        other: VectorProtocol,
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        """Like ``np.ndarray.allclose``, but for MomentumArray4D."""
        return ak.all(self.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan))


behavior["*", "Momentum2D"] = MomentumArray2D


class MomentumRecord2D(MomentumAwkward2D, ak.Record):  # type: ignore[misc]
    """
    Defines ``awkward`` behavior for a 2 dimensional momentum record.

    See :class:`MomentumArray2D` and :class:`VectorArray2D`
    for the corresponding ``Momentum`` and ``Vector`` classes.
    """


behavior["Momentum2D"] = MomentumRecord2D


class MomentumArray3D(MomentumAwkward3D, ak.Array):  # type: ignore[misc]
    """
    Defines ``awkward`` behavior for a 3 dimensional momentum vector.

    See :class:`MomentumRecord3D` and :class:`VectorArray3D`
    for the corresponding ``Record`` and ``Vector`` classes.
    """

    def allclose(
        self,
        other: VectorProtocol,
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        return ak.all(self.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan))


behavior["*", "Momentum3D"] = MomentumArray3D


class MomentumRecord3D(MomentumAwkward3D, ak.Record):  # type: ignore[misc]
    """
    Defines ``awkward`` behavior for a 3 dimensional momentum record.

    See :class:`MomentumArray3D` and :class:`VectorArray3D`
    for the corresponding ``Momentum`` and ``Vector`` classes.
    """


behavior["Momentum3D"] = MomentumRecord3D


class MomentumArray4D(MomentumAwkward4D, ak.Array):  # type: ignore[misc]
    """
    Defines ``awkward`` behavior for a 4 dimensional momentum vector.

    See :class:`MomentumRecord4D` and :class:`VectorArray4D`
    for the corresponding ``Record`` and ``Vector`` classes.
    """

    def allclose(
        self,
        other: VectorProtocol,
        rtol: ScalarCollection = 1e-05,
        atol: ScalarCollection = 1e-08,
        equal_nan: BoolCollection = False,
    ) -> BoolCollection:
        return ak.all(self.isclose(other, rtol=rtol, atol=atol, equal_nan=equal_nan))


behavior["*", "Momentum4D"] = MomentumArray4D


class MomentumRecord4D(MomentumAwkward4D, ak.Record):  # type: ignore[misc]
    """
    Defines ``awkward`` behavior for a 4 dimensional momentum record.

    See :class:`MomentumArray4D` and :class:`VectorArray4D`
    for the corresponding ``Momentum`` and ``Vector`` classes.
    """


behavior["Momentum4D"] = MomentumRecord4D

# NumPy functions, which also affect operator overloading #####################

behavior[numpy.absolute, "Vector2D"] = lambda v: v.rho
behavior[numpy.absolute, "Vector3D"] = lambda v: v.mag
behavior[numpy.absolute, "Vector4D"] = lambda v: v.tau
behavior[numpy.absolute, "Momentum2D"] = lambda v: v.rho
behavior[numpy.absolute, "Momentum3D"] = lambda v: v.mag
behavior[numpy.absolute, "Momentum4D"] = lambda v: v.tau

behavior[numpy.square, "Vector2D"] = lambda v: v.rho2
behavior[numpy.square, "Vector3D"] = lambda v: v.mag2
behavior[numpy.square, "Vector4D"] = lambda v: v.tau2
behavior[numpy.square, "Momentum2D"] = lambda v: v.rho2
behavior[numpy.square, "Momentum3D"] = lambda v: v.mag2
behavior[numpy.square, "Momentum4D"] = lambda v: v.tau2

behavior[numpy.sqrt, "Vector2D"] = lambda v: v.rho2**0.25
behavior[numpy.sqrt, "Vector3D"] = lambda v: v.mag2**0.25
behavior[numpy.sqrt, "Vector4D"] = lambda v: v.tau2**0.25
behavior[numpy.sqrt, "Momentum2D"] = lambda v: v.rho2**0.25
behavior[numpy.sqrt, "Momentum3D"] = lambda v: v.mag2**0.25
behavior[numpy.sqrt, "Momentum4D"] = lambda v: v.tau2**0.25

behavior[numpy.cbrt, "Vector2D"] = lambda v: v.rho2**0.16666666666666666
behavior[numpy.cbrt, "Vector3D"] = lambda v: v.mag2**0.16666666666666666
behavior[numpy.cbrt, "Vector4D"] = lambda v: v.tau2**0.16666666666666666
behavior[numpy.cbrt, "Momentum2D"] = lambda v: v.rho2**0.16666666666666666
behavior[numpy.cbrt, "Momentum3D"] = lambda v: v.mag2**0.16666666666666666
behavior[numpy.cbrt, "Momentum4D"] = lambda v: v.tau2**0.16666666666666666

behavior[numpy.power, "Vector2D", numbers.Real] = (
    lambda v, expo: v.rho2 if expo == 2 else v.rho**expo
)
behavior[numpy.power, "Vector3D", numbers.Real] = (
    lambda v, expo: v.mag2 if expo == 2 else v.mag**expo
)
behavior[numpy.power, "Vector4D", numbers.Real] = (
    lambda v, expo: v.tau2 if expo == 2 else v.tau**expo
)
behavior[numpy.power, "Momentum2D", numbers.Real] = (
    lambda v, expo: v.rho2 if expo == 2 else v.rho**expo
)
behavior[numpy.power, "Momentum3D", numbers.Real] = (
    lambda v, expo: v.mag2 if expo == 2 else v.mag**expo
)
behavior[numpy.power, "Momentum4D", numbers.Real] = (
    lambda v, expo: v.tau2 if expo == 2 else v.tau**expo
)

behavior["__cast__", VectorNumpy2D] = lambda v: vector.Array(v)
behavior["__cast__", VectorNumpy3D] = lambda v: vector.Array(v)
behavior["__cast__", VectorNumpy4D] = lambda v: vector.Array(v)

for left in (
    "Vector2D",
    "Vector3D",
    "Vector4D",
    "Momentum2D",
    "Momentum3D",
    "Momentum4D",
    VectorObject2D,
    VectorObject3D,
    VectorObject4D,
):
    for right in (
        "Vector2D",
        "Vector3D",
        "Vector4D",
        "Momentum2D",
        "Momentum3D",
        "Momentum4D",
        VectorObject2D,
        VectorObject3D,
        VectorObject4D,
    ):
        if not (isinstance(left, type) and isinstance(right, type)):
            behavior[numpy.add, left, right] = lambda v1, v2: v1.add(v2)
            behavior[numpy.subtract, left, right] = lambda v1, v2: v1.subtract(v2)
            behavior[numpy.matmul, left, right] = lambda v1, v2: v1.dot(v2)
            behavior[numpy.equal, left, right] = lambda v1, v2: v1.equal(v2)
            behavior[numpy.not_equal, left, right] = lambda v1, v2: v1.not_equal(v2)

for name in (
    "Vector2D",
    "Vector3D",
    "Vector4D",
    "Momentum2D",
    "Momentum3D",
    "Momentum4D",
):
    behavior[numpy.multiply, name, numbers.Real] = lambda v, factor: v.scale(factor)
    behavior[numpy.multiply, numbers.Real, name] = lambda factor, v: v.scale(factor)
    behavior[numpy.negative, name] = lambda v: v.scale(-1)
    behavior[numpy.positive, name] = lambda v: v
    behavior[numpy.true_divide, name, numbers.Real] = lambda v, denom: v.scale(
        1 / denom
    )

# class object cross-references ###############################################

VectorArray2D.ProjectionClass2D = VectorArray2D
VectorArray2D.ProjectionClass3D = VectorArray3D
VectorArray2D.ProjectionClass4D = VectorArray4D
VectorArray2D.GenericClass = VectorArray2D
VectorArray2D.MomentumClass = MomentumArray2D

VectorRecord2D.ProjectionClass2D = VectorRecord2D
VectorRecord2D.ProjectionClass3D = VectorRecord3D
VectorRecord2D.ProjectionClass4D = VectorRecord4D
VectorRecord2D.GenericClass = VectorRecord2D
VectorRecord2D.MomentumClass = MomentumRecord2D

MomentumArray2D.ProjectionClass2D = MomentumArray2D
MomentumArray2D.ProjectionClass3D = MomentumArray3D
MomentumArray2D.ProjectionClass4D = MomentumArray4D
MomentumArray2D.GenericClass = VectorArray2D
MomentumArray2D.MomentumClass = MomentumArray2D

MomentumRecord2D.ProjectionClass2D = MomentumRecord2D
MomentumRecord2D.ProjectionClass3D = MomentumRecord3D
MomentumRecord2D.ProjectionClass4D = MomentumRecord4D
MomentumRecord2D.GenericClass = VectorRecord2D
MomentumRecord2D.MomentumClass = MomentumRecord2D

VectorArray3D.ProjectionClass2D = VectorArray2D
VectorArray3D.ProjectionClass3D = VectorArray3D
VectorArray3D.ProjectionClass4D = VectorArray4D
VectorArray3D.GenericClass = VectorArray3D
VectorArray3D.MomentumClass = MomentumArray3D

VectorRecord3D.ProjectionClass2D = VectorRecord2D
VectorRecord3D.ProjectionClass3D = VectorRecord3D
VectorRecord3D.ProjectionClass4D = VectorRecord4D
VectorRecord3D.GenericClass = VectorRecord3D
VectorRecord3D.MomentumClass = MomentumRecord3D

MomentumArray3D.ProjectionClass2D = MomentumArray2D
MomentumArray3D.ProjectionClass3D = MomentumArray3D
MomentumArray3D.ProjectionClass4D = MomentumArray4D
MomentumArray3D.GenericClass = VectorArray3D
MomentumArray3D.MomentumClass = MomentumArray3D

MomentumRecord3D.ProjectionClass2D = MomentumRecord2D
MomentumRecord3D.ProjectionClass3D = MomentumRecord3D
MomentumRecord3D.ProjectionClass4D = MomentumRecord4D
MomentumRecord3D.GenericClass = VectorRecord3D
MomentumRecord3D.MomentumClass = MomentumRecord3D

VectorArray4D.ProjectionClass2D = VectorArray2D
VectorArray4D.ProjectionClass3D = VectorArray3D
VectorArray4D.ProjectionClass4D = VectorArray4D
VectorArray4D.GenericClass = VectorArray4D
VectorArray4D.MomentumClass = MomentumArray4D

VectorRecord4D.ProjectionClass2D = VectorRecord2D
VectorRecord4D.ProjectionClass3D = VectorRecord3D
VectorRecord4D.ProjectionClass4D = VectorRecord4D
VectorRecord4D.GenericClass = VectorRecord4D
VectorRecord4D.MomentumClass = MomentumRecord4D

MomentumArray4D.ProjectionClass2D = MomentumArray2D
MomentumArray4D.ProjectionClass3D = MomentumArray3D
MomentumArray4D.ProjectionClass4D = MomentumArray4D
MomentumArray4D.GenericClass = VectorArray4D
MomentumArray4D.MomentumClass = MomentumArray4D

MomentumRecord4D.ProjectionClass2D = MomentumRecord2D
MomentumRecord4D.ProjectionClass3D = MomentumRecord3D
MomentumRecord4D.ProjectionClass4D = MomentumRecord4D
MomentumRecord4D.GenericClass = VectorRecord4D
MomentumRecord4D.MomentumClass = MomentumRecord4D


# implementation of behaviors in Numba ########################################

if vector._is_awkward_v2:

    def _lookup_field(record_type: typing.Any, name: str) -> int:
        return record_type.fields.index(name)

else:

    def _lookup_field(record_type: typing.Any, name: str) -> int:
        return record_type.recordlookup.index(name)


def _arraytype_of(awkwardtype: typing.Any, component: str) -> typing.Any:
    import numba

    if isinstance(
        awkwardtype,
        ak._connect.numba.layout.NumpyArrayType
        if hasattr(ak._connect, "numba")  # Awkward v2
        else ak._connect._numba.layout.NumpyArrayType,
    ):
        return awkwardtype.arraytype
    elif isinstance(
        awkwardtype,
        ak._connect.numba.layout.IndexedArrayType
        if hasattr(ak._connect, "numba")  # Awkward v2
        else ak._connect._numba.layout.IndexedArrayType,
    ):
        return _arraytype_of(awkwardtype.contenttype, component)
    raise numba.TypingError(
        f"vector components like {component!r} must be NumpyArrayType, not {awkwardtype}"
    )


def _aztype_of(recordarraytype: typing.Any, is_momentum: bool) -> typing.Any:
    import numba

    cls: type[AzimuthalObjectXY] | type[AzimuthalObjectRhoPhi]

    x_index = None
    y_index = None
    rho_index = None
    phi_index = None

    if is_momentum:
        try:
            x_index = _lookup_field(recordarraytype, "px")
        except ValueError:
            x_index = None
    if x_index is None:
        try:
            x_index = _lookup_field(recordarraytype, "x")
        except ValueError:
            x_index = None
    if is_momentum:
        try:
            y_index = _lookup_field(recordarraytype, "py")
        except ValueError:
            y_index = None
    if y_index is None:
        try:
            y_index = _lookup_field(recordarraytype, "y")
        except ValueError:
            y_index = None
    if is_momentum:
        try:
            rho_index = _lookup_field(recordarraytype, "pt")
        except ValueError:
            rho_index = None
    if rho_index is None:
        try:
            rho_index = _lookup_field(recordarraytype, "rho")
        except ValueError:
            rho_index = None
    try:
        phi_index = _lookup_field(recordarraytype, "phi")
    except ValueError:
        phi_index = None

    if x_index is not None and y_index is not None:
        coord1 = _arraytype_of(recordarraytype.contenttypes[x_index], "x").dtype
        coord2 = _arraytype_of(recordarraytype.contenttypes[y_index], "y").dtype
        cls = AzimuthalObjectXY

    elif rho_index is not None and phi_index is not None:
        coord1 = _arraytype_of(recordarraytype.contenttypes[rho_index], "rho").dtype
        coord2 = _arraytype_of(recordarraytype.contenttypes[phi_index], "phi").dtype
        cls = AzimuthalObjectRhoPhi

    elif is_momentum:
        raise numba.TypingError(
            f"{recordarraytype} is missing azimuthal fields: px/py (x/y) or pt/phi (rho/phi)"
        )

    else:
        raise numba.TypingError(
            f"{recordarraytype} is missing azimuthal fields: x/y or rho/phi"
        )

    return numba.typeof(cls(coord1.cast_python_value(0), coord2.cast_python_value(0)))


def _ltype_of(recordarraytype: typing.Any, is_momentum: bool) -> typing.Any:
    import numba

    cls: (
        type[LongitudinalObjectZ]
        | type[LongitudinalObjectTheta]
        | type[LongitudinalObjectEta]
    )

    z_index = None
    theta_index = None
    eta_index = None

    if is_momentum:
        try:
            z_index = _lookup_field(recordarraytype, "pz")
        except ValueError:
            z_index = None
    if z_index is None:
        try:
            z_index = _lookup_field(recordarraytype, "z")
        except ValueError:
            z_index = None
    try:
        theta_index = _lookup_field(recordarraytype, "theta")
    except ValueError:
        theta_index = None
    try:
        eta_index = _lookup_field(recordarraytype, "eta")
    except ValueError:
        eta_index = None

    if z_index is not None:
        coord1 = _arraytype_of(recordarraytype.contenttypes[z_index], "z").dtype
        cls = LongitudinalObjectZ

    elif theta_index is not None:
        coord1 = _arraytype_of(recordarraytype.contenttypes[theta_index], "theta").dtype
        cls = LongitudinalObjectTheta

    elif eta_index is not None:
        coord1 = _arraytype_of(recordarraytype.contenttypes[eta_index], "eta").dtype
        cls = LongitudinalObjectEta

    elif is_momentum:
        raise numba.TypingError(
            f"{recordarraytype} is missing longitudinal fields: pz (z) or theta or eta"
        )

    else:
        raise numba.TypingError(
            f"{recordarraytype} is missing longitudinal fields: z or theta or eta"
        )

    return numba.typeof(cls(coord1.cast_python_value(0)))


def _ttype_of(recordarraytype: typing.Any, is_momentum: bool) -> typing.Any:
    import numba

    cls: type[TemporalObjectT] | type[TemporalObjectTau]

    t_index = None
    tau_index = None

    if is_momentum:
        try:
            t_index = _lookup_field(recordarraytype, "E")
        except ValueError:
            t_index = None
    if is_momentum and t_index is None:
        try:
            t_index = _lookup_field(recordarraytype, "e")
        except ValueError:
            t_index = None
    if is_momentum and t_index is None:
        try:
            t_index = _lookup_field(recordarraytype, "energy")
        except ValueError:
            t_index = None
    if t_index is None:
        try:
            t_index = _lookup_field(recordarraytype, "t")
        except ValueError:
            t_index = None
    if is_momentum:
        try:
            tau_index = _lookup_field(recordarraytype, "M")
        except ValueError:
            tau_index = None
    if is_momentum and tau_index is None:
        try:
            tau_index = _lookup_field(recordarraytype, "m")
        except ValueError:
            tau_index = None
    if is_momentum and tau_index is None:
        try:
            tau_index = _lookup_field(recordarraytype, "mass")
        except ValueError:
            tau_index = None
    if tau_index is None:
        try:
            tau_index = _lookup_field(recordarraytype, "tau")
        except ValueError:
            tau_index = None

    if t_index is not None:
        coord1 = _arraytype_of(recordarraytype.contenttypes[t_index], "t").dtype
        cls = TemporalObjectT

    elif tau_index is not None:
        coord1 = _arraytype_of(recordarraytype.contenttypes[tau_index], "tau").dtype
        cls = TemporalObjectTau

    elif is_momentum:
        raise numba.TypingError(
            f"{recordarraytype} is missing temporal fields: E/e/energy (t) or M/m/mass (tau)"
        )

    else:
        raise numba.TypingError(
            f"{recordarraytype} is missing temporal fields: t or tau"
        )

    return numba.typeof(cls(coord1.cast_python_value(0)))


def _numba_typer_Vector2D(viewtype: typing.Any) -> typing.Any:
    import vector.backends._numba_object

    # These exist, but the file is type ignored, so can't be found
    return vector.backends._numba_object.VectorObject2DType(  # type: ignore[attr-defined]
        _aztype_of(viewtype.arrayviewtype.type, False)
    )


def _numba_typer_Vector3D(viewtype: typing.Any) -> typing.Any:
    import vector.backends._numba_object

    return vector.backends._numba_object.VectorObject3DType(  # type: ignore[attr-defined]
        _aztype_of(viewtype.arrayviewtype.type, False),
        _ltype_of(viewtype.arrayviewtype.type, False),
    )


def _numba_typer_Vector4D(viewtype: typing.Any) -> typing.Any:
    import vector.backends._numba_object

    return vector.backends._numba_object.VectorObject4DType(  # type: ignore[attr-defined]
        _aztype_of(viewtype.arrayviewtype.type, False),
        _ltype_of(viewtype.arrayviewtype.type, False),
        _ttype_of(viewtype.arrayviewtype.type, False),
    )


def _numba_typer_Momentum2D(viewtype: typing.Any) -> typing.Any:
    import vector.backends._numba_object

    return vector.backends._numba_object.MomentumObject2DType(  # type: ignore[attr-defined]
        _aztype_of(viewtype.arrayviewtype.type, True)
    )


def _numba_typer_Momentum3D(viewtype: typing.Any) -> typing.Any:
    import vector.backends._numba_object

    return vector.backends._numba_object.MomentumObject3DType(  # type: ignore[attr-defined]
        _aztype_of(viewtype.arrayviewtype.type, True),
        _ltype_of(viewtype.arrayviewtype.type, True),
    )


def _numba_typer_Momentum4D(viewtype: typing.Any) -> typing.Any:
    import vector.backends._numba_object

    return vector.backends._numba_object.MomentumObject4DType(  # type: ignore[attr-defined]
        _aztype_of(viewtype.arrayviewtype.type, True),
        _ltype_of(viewtype.arrayviewtype.type, True),
        _ttype_of(viewtype.arrayviewtype.type, True),
    )


def _numba_lower(
    context: typing.Any, builder: typing.Any, sig: typing.Any, args: typing.Any
) -> typing.Any:
    from vector.backends._numba_object import (  # type: ignore[attr-defined]
        _awkward_numba_E,
        _awkward_numba_e,
        _awkward_numba_energy,
        _awkward_numba_eta,
        _awkward_numba_M,
        _awkward_numba_m,
        _awkward_numba_mass,
        _awkward_numba_ptphi,
        _awkward_numba_pxpy,
        _awkward_numba_pxy,
        _awkward_numba_pz,
        _awkward_numba_rhophi,
        _awkward_numba_t,
        _awkward_numba_tau,
        _awkward_numba_theta,
        _awkward_numba_xpy,
        _awkward_numba_xy,
        _awkward_numba_z,
    )

    vectorcls = sig.return_type.instance_class

    fields = (
        sig.args[0].arrayviewtype.type.fields
        if vector._is_awkward_v2
        else sig.args[0].arrayviewtype.type.recordlookup
    )

    if issubclass(vectorcls, (VectorObject2D, VectorObject3D, VectorObject4D)):
        if issubclass(sig.return_type.azimuthaltype.instance_class, AzimuthalXY):
            if "x" in fields and "y" in fields:
                azimuthal = _awkward_numba_xy
            elif "x" in fields and "py" in fields:
                azimuthal = _awkward_numba_xpy
            elif "px" in fields and "y" in fields:
                azimuthal = _awkward_numba_pxy
            elif "px" in fields and "py" in fields:
                azimuthal = _awkward_numba_pxpy
            else:
                raise AssertionError
        elif issubclass(sig.return_type.azimuthaltype.instance_class, AzimuthalRhoPhi):
            if "rho" in fields and "phi" in fields:
                azimuthal = _awkward_numba_rhophi
            elif "pt" in fields and "phi" in fields:
                azimuthal = _awkward_numba_ptphi
            else:
                raise AssertionError

    if issubclass(vectorcls, (VectorObject3D, VectorObject4D)):
        if issubclass(sig.return_type.longitudinaltype.instance_class, LongitudinalZ):
            if "z" in fields:
                longitudinal = _awkward_numba_z
            elif "pz" in fields:
                longitudinal = _awkward_numba_pz
            else:
                raise AssertionError
        elif issubclass(
            sig.return_type.longitudinaltype.instance_class, LongitudinalTheta
        ):
            longitudinal = _awkward_numba_theta
        elif issubclass(
            sig.return_type.longitudinaltype.instance_class, LongitudinalEta
        ):
            longitudinal = _awkward_numba_eta

    if issubclass(vectorcls, VectorObject4D):
        if issubclass(sig.return_type.temporaltype.instance_class, TemporalT):
            if "t" in fields:
                temporal = _awkward_numba_t
            elif "E" in fields:
                temporal = _awkward_numba_E
            elif "e" in fields:
                temporal = _awkward_numba_e
            elif "energy" in fields:
                temporal = _awkward_numba_energy
            else:
                raise AssertionError
        elif issubclass(sig.return_type.temporaltype.instance_class, TemporalTau):
            if "tau" in fields:
                temporal = _awkward_numba_tau
            elif "M" in fields:
                temporal = _awkward_numba_M
            elif "m" in fields:
                temporal = _awkward_numba_m
            elif "mass" in fields:
                temporal = _awkward_numba_mass
            else:
                raise AssertionError

    if issubclass(vectorcls, VectorObject2D):

        def impl(record: typing.Any) -> typing.Any:
            return vectorcls(azimuthal(record))

    elif issubclass(vectorcls, VectorObject3D):

        def impl(record: typing.Any) -> typing.Any:
            return vectorcls(azimuthal(record), longitudinal(record))

    elif issubclass(vectorcls, VectorObject4D):

        def impl(record: typing.Any) -> typing.Any:
            return vectorcls(azimuthal(record), longitudinal(record), temporal(record))

    return context.compile_internal(builder, impl, sig, args)


behavior["__numba_typer__", "Vector2D"] = _numba_typer_Vector2D
behavior["__numba_typer__", "Vector3D"] = _numba_typer_Vector3D
behavior["__numba_typer__", "Vector4D"] = _numba_typer_Vector4D
behavior["__numba_typer__", "Momentum2D"] = _numba_typer_Momentum2D
behavior["__numba_typer__", "Momentum3D"] = _numba_typer_Momentum3D
behavior["__numba_typer__", "Momentum4D"] = _numba_typer_Momentum4D

behavior["__numba_lower__", "Vector2D"] = _numba_lower
behavior["__numba_lower__", "Vector3D"] = _numba_lower
behavior["__numba_lower__", "Vector4D"] = _numba_lower
behavior["__numba_lower__", "Momentum2D"] = _numba_lower
behavior["__numba_lower__", "Momentum3D"] = _numba_lower
behavior["__numba_lower__", "Momentum4D"] = _numba_lower


def _reduce_sum(
    array: VectorArray2D
    | VectorArray3D
    | VectorArray4D
    | MomentumArray2D
    | MomentumArray3D
    | MomentumArray4D,
    mask_identity: bool,
) -> VectorProtocol:
    fields = {}
    if isinstance(array, Lorentz):
        fields["t"] = numpy.sum(array.t, axis=1)
    if isinstance(array, Spatial):
        fields["z"] = numpy.sum(array.z, axis=1)

    assert isinstance(array, Planar)
    fields["x"] = numpy.sum(array.x, axis=1)
    fields["y"] = numpy.sum(array.y, axis=1)

    layout = ak.to_layout(array)

    return ak.zip(
        fields,
        behavior=array.behavior,
        with_name=layout.purelist_parameter("__record__"),
    )


def _reduce_count(
    array: VectorArray2D
    | VectorArray3D
    | VectorArray4D
    | MomentumArray2D
    | MomentumArray3D
    | MomentumArray4D,
    mask_identity: bool,
) -> VectorProtocol:
    first_field = array[array.fields[0]]
    return ak.count(first_field, axis=1)


def _reduce_count_nonzero(
    array: VectorArray2D
    | VectorArray3D
    | VectorArray4D
    | MomentumArray2D
    | MomentumArray3D
    | MomentumArray4D,
    mask_identity: bool,
) -> ScalarCollection:
    is_nonzero = array.rho2 != 0
    if isinstance(array, Spatial):
        is_nonzero = numpy.logical_or(is_nonzero, array.z != 0)
    if isinstance(array, Lorentz):
        is_nonzero = numpy.logical_or(is_nonzero, array.t2 != 0)

    return ak.count_nonzero(is_nonzero, axis=1)


for reducer, impl in (
    (ak.sum, _reduce_sum),
    (ak.count, _reduce_count),
    (ak.count_nonzero, _reduce_count_nonzero),
):
    for dim in range(2, 5):
        behavior[reducer, f"Vector{dim}D"] = impl
        behavior[reducer, f"Momentum{dim}D"] = impl
