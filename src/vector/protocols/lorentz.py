# -*- coding: utf-8 -*-
from __future__ import annotations  # type: ignore

from typing import Protocol, Tuple, overload

Scalar = float
LorentzTuple = Tuple[Scalar, Scalar, Scalar, Scalar]


class LorentzVector(Protocol):
    def __init__(self, x: Scalar, y: Scalar, z: Scalar, t: Scalar):
        ...

    x: Scalar
    y: Scalar
    z: Scalar
    t: Scalar

    pt: Scalar
    eta: Scalar
    phi: Scalar

    mag: Scalar
    mag2: Scalar

    def __add__(self, other: LorentzVector) -> LorentzVector:
        ...

    @overload
    def __mul__(self, other: Scalar) -> LorentzVector:
        ...

    @overload
    def __mul__(self, other: LorentzVector) -> Scalar:
        ...

    def __mul__(self, other):
        ...
