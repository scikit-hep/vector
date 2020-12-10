# -*- coding: utf-8 -*-
from __future__ import annotations  # type: ignore

from typing import Protocol, Tuple, TypeVar, overload

V = TypeVar("V", covariant=True)
T = TypeVar("T")

GenericLorentzTuple = Tuple[V, V, V, V]


Self = TypeVar("Self", bound="Scalar")


class Scalar(Protocol):
    def __add__(self: Self, other: Self) -> Self:
        ...

    def __radd__(self: Self, other: Self) -> Self:
        ...

    def __sub__(self: Self, other: Self) -> Self:
        ...

    def __rsub__(self: Self, other: Self) -> Self:
        ...

    def __mul__(self: Self, other: Self) -> Self:
        ...

    def __rmul__(self: Self, other: Self) -> Self:
        ...

    def __truediv__(self: Self, other: Self) -> Self:
        ...

    def __pow__(self: Self, other: Self) -> Self:
        ...


class GenericLorentzVector(Protocol[T]):
    def __init__(self, x: T, y: T, z: T, t: T):
        ...

    @property
    def x(self) -> T:
        ...

    @property
    def y(self) -> T:
        ...

    @property
    def z(self) -> T:
        ...

    @property
    def t(self) -> T:
        ...

    @property
    def pt(self) -> T:
        ...

    @property
    def eta(self) -> T:
        ...

    @property
    def phi(self) -> T:
        ...

    @property
    def mag(self) -> T:
        ...

    @property
    def mag2(self) -> T:
        ...

    def __add__(self, other: GenericLorentzVector) -> GenericLorentzVector:
        ...

    @overload
    def __mul__(self, other: T) -> GenericLorentzVector:
        ...

    @overload
    def __mul__(self, other: GenericLorentzVector) -> T:
        ...

    def __mul__(self, other):
        ...


LorentzVector = GenericLorentzVector[Scalar]
LorentzTuple = GenericLorentzTuple[Scalar]
