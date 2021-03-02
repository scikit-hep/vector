from typing import Protocol, Tuple, TypeVar, overload

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


LorentzTuple = Tuple[Scalar, Scalar, Scalar, Scalar]


class LorentzVector(Protocol):
    def __init__(self, x: Scalar, y: Scalar, z: Scalar, t: Scalar):
        ...

    @property
    def x(self) -> Scalar:
        ...

    @property
    def y(self) -> Scalar:
        ...

    @property
    def z(self) -> Scalar:
        ...

    @property
    def t(self) -> Scalar:
        ...

    @property
    def pt(self) -> Scalar:
        ...

    @property
    def eta(self) -> Scalar:
        ...

    @property
    def phi(self) -> Scalar:
        ...

    @property
    def mag(self) -> Scalar:
        ...

    @property
    def mag2(self) -> Scalar:
        ...

    @overload
    def __add__(self, other: "LorentzVector") -> "LorentzVector":
        ...

    @overload
    def __add__(self, other: Scalar) -> "LorentzVector":
        ...

    def __add__(self, other):
        ...

    @overload
    def __mul__(self, other: "LorentzVector") -> Scalar:
        ...

    @overload
    def __mul__(self, other: Scalar) -> "LorentzVector":
        ...

    def __mul__(self, other):
        ...
