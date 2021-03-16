# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.


class Vector:
    pass


class Vector2D(Vector):
    def to_xy(self):
        "to_xy docs"
        from .compute.planar import x, y

        return self._wrap_result((x.dispatch(self), y.dispatch(self)), [AzimuthalXY])

    def to_rhophi(self):
        "to_rhophi docs"
        from .compute.planar import phi, rho

        return self._wrap_result(
            (rho.dispatch(self), phi.dispatch(self)), [AzimuthalRhoPhi]
        )


class Vector3D(Vector):
    def to_Vector2D(self):
        "to_Vector2D docs"
        return self.ProjectionClass2D._wrap_result(
            self.ProjectionClass2D, self.azimuthal.elements, [aztype(self)]
        )

    def to_xy(self):
        "to_xy docs, mention projection"
        return self.to_Vector2D().to_xy()

    def to_rhophi(self):
        "to_rhophi docs, mention projection"
        return self.to_Vector2D().to_rhophi()

    def to_xyz(self):
        "to_xyz docs"
        from .compute.planar import x, y
        from .compute.spatial import z

        return self._wrap_result(
            (x.dispatch(self), y.dispatch(self), z.dispatch(self)),
            [AzimuthalXY, LongitudinalZ],
        )

    def to_xytheta(self):
        "to_xytheta docs"
        from .compute.planar import x, y
        from .compute.spatial import theta

        return self._wrap_result(
            (x.dispatch(self), y.dispatch(self), theta.dispatch(self)),
            [AzimuthalXY, LongitudinalTheta],
        )

    def to_xyeta(self):
        "to_xyeta docs"
        from .compute.planar import x, y
        from .compute.spatial import eta

        return self._wrap_result(
            (x.dispatch(self), y.dispatch(self), eta.dispatch(self)),
            [AzimuthalXY, LongitudinalEta],
        )

    def to_rhophiz(self):
        "to_rhophiz docs"
        from .compute.planar import phi, rho
        from .compute.spatial import z

        return self._wrap_result(
            (rho.dispatch(self), phi.dispatch(self), z.dispatch(self)),
            [AzimuthalRhoPhi, LongitudinalZ],
        )

    def to_rhophitheta(self):
        "to_rhophitheta docs"
        from .compute.planar import phi, rho
        from .compute.spatial import theta

        return self._wrap_result(
            (rho.dispatch(self), phi.dispatch(self), theta.dispatch(self)),
            [AzimuthalRhoPhi, LongitudinalTheta],
        )

    def to_rhophieta(self):
        "to_rhophieta docs"
        from .compute.planar import phi, rho
        from .compute.spatial import eta

        return self._wrap_result(
            (rho.dispatch(self), phi.dispatch(self), eta.dispatch(self)),
            [AzimuthalRhoPhi, LongitudinalEta],
        )


class Vector4D(Vector):
    def to_Vector3D(self):
        "to_Vector3D docs"
        return self.ProjectionClass3D._wrap_result(
            self.ProjectionClass3D,
            self.azimuthal.elements + self.longitudinal.elements,
            [aztype(self), ltype(self)],
        )

    def to_xyz(self):
        "to_xyz docs, mention projection"
        return self.to_Vector3D().to_xyz()

    def to_xytheta(self):
        "to_xytheta docs, mention projection"
        return self.to_Vector3D().to_xytheta()

    def to_xyeta(self):
        "to_xyeta docs, mention projection"
        return self.to_Vector3D().to_xyeta()

    def to_rhophiz(self):
        "to_rhophiz docs, mention projection"
        return self.to_Vector3D().to_rhophiz()

    def to_rhophitheta(self):
        "to_rhophitheta docs, mention projection"
        return self.to_Vector3D().to_rhophitheta()

    def to_rhophieta(self):
        "to_rhophieta docs, mention projection"
        return self.to_Vector3D().to_rhophieta()

    def to_xyzt(self):
        "to_xyzt docs"
        from .compute.lorentz import t
        from .compute.planar import x, y
        from .compute.spatial import z

        return self._wrap_result(
            (x.dispatch(self), y.dispatch(self), z.dispatch(self), t.dispatch(self)),
            [AzimuthalXY, LongitudinalZ, TemporalT],
        )

    def to_xyztau(self):
        "to_xyztau docs"
        from .compute.lorentz import tau
        from .compute.planar import x, y
        from .compute.spatial import z

        return self._wrap_result(
            (x.dispatch(self), y.dispatch(self), z.dispatch(self), tau.dispatch(self)),
            [AzimuthalXY, LongitudinalZ, TemporalTau],
        )

    def to_xythetat(self):
        "to_xythetat docs"
        from .compute.lorentz import t
        from .compute.planar import x, y
        from .compute.spatial import theta

        return self._wrap_result(
            (
                x.dispatch(self),
                y.dispatch(self),
                theta.dispatch(self),
                t.dispatch(self),
            ),
            [AzimuthalXY, LongitudinalTheta, TemporalT],
        )

    def to_xythetatau(self):
        "to_xythetatau docs"
        from .compute.lorentz import tau
        from .compute.planar import x, y
        from .compute.spatial import theta

        return self._wrap_result(
            (
                x.dispatch(self),
                y.dispatch(self),
                theta.dispatch(self),
                tau.dispatch(self),
            ),
            [AzimuthalXY, LongitudinalTheta, TemporalTau],
        )

    def to_xyetat(self):
        "to_xyetat docs"
        from .compute.lorentz import t
        from .compute.planar import x, y
        from .compute.spatial import eta

        return self._wrap_result(
            (x.dispatch(self), y.dispatch(self), eta.dispatch(self), t.dispatch(self)),
            [AzimuthalXY, LongitudinalEta, TemporalT],
        )

    def to_xyetatau(self):
        "to_xyetatau docs"
        from .compute.lorentz import tau
        from .compute.planar import x, y
        from .compute.spatial import eta

        return self._wrap_result(
            (
                x.dispatch(self),
                y.dispatch(self),
                eta.dispatch(self),
                tau.dispatch(self),
            ),
            [AzimuthalXY, LongitudinalEta, TemporalTau],
        )

    def to_rhophizt(self):
        "to_rhophizt docs"
        from .compute.lorentz import t
        from .compute.planar import phi, rho
        from .compute.spatial import z

        return self._wrap_result(
            (
                rho.dispatch(self),
                phi.dispatch(self),
                z.dispatch(self),
                t.dispatch(self),
            ),
            [AzimuthalRhoPhi, LongitudinalZ, TemporalT],
        )

    def to_rhophiztau(self):
        "to_rhophiztau docs"
        from .compute.lorentz import tau
        from .compute.planar import phi, rho
        from .compute.spatial import z

        return self._wrap_result(
            (
                rho.dispatch(self),
                phi.dispatch(self),
                z.dispatch(self),
                tau.dispatch(self),
            ),
            [AzimuthalRhoPhi, LongitudinalZ, TemporalTau],
        )

    def to_rhophithetat(self):
        "to_rhophithetat docs"
        from .compute.lorentz import t
        from .compute.planar import phi, rho
        from .compute.spatial import theta

        return self._wrap_result(
            (
                rho.dispatch(self),
                phi.dispatch(self),
                theta.dispatch(self),
                t.dispatch(self),
            ),
            [AzimuthalRhoPhi, LongitudinalTheta, TemporalT],
        )

    def to_rhophithetatau(self):
        "to_rhophithetatau docs"
        from .compute.lorentz import tau
        from .compute.planar import phi, rho
        from .compute.spatial import theta

        return self._wrap_result(
            (
                rho.dispatch(self),
                phi.dispatch(self),
                theta.dispatch(self),
                tau.dispatch(self),
            ),
            [AzimuthalRhoPhi, LongitudinalTheta, TemporalTau],
        )

    def to_rhophietat(self):
        "to_rhophietat docs"
        from .compute.lorentz import t
        from .compute.planar import phi, rho
        from .compute.spatial import eta

        return self._wrap_result(
            (
                rho.dispatch(self),
                phi.dispatch(self),
                eta.dispatch(self),
                t.dispatch(self),
            ),
            [AzimuthalRhoPhi, LongitudinalEta, TemporalT],
        )

    def to_rhophietatau(self):
        "to_rhophietatau docs"
        from .compute.lorentz import tau
        from .compute.planar import phi, rho
        from .compute.spatial import eta

        return self._wrap_result(
            (
                rho.dispatch(self),
                phi.dispatch(self),
                eta.dispatch(self),
                tau.dispatch(self),
            ),
            [AzimuthalRhoPhi, LongitudinalEta, TemporalTau],
        )


class Coordinates:
    pass


class Azimuthal(Coordinates):
    @property
    def elements(self):
        "azimuthal elements docs"
        raise AssertionError


class Longitudinal(Coordinates):
    @property
    def elements(self):
        "longitudinal elements docs"
        raise AssertionError


class Temporal(Coordinates):
    @property
    def elements(self):
        "temporal elements docs"
        raise AssertionError


class AzimuthalXY(Azimuthal):
    pass


class AzimuthalRhoPhi(Azimuthal):
    pass


class LongitudinalZ(Longitudinal):
    pass


class LongitudinalTheta(Longitudinal):
    pass


class LongitudinalEta(Longitudinal):
    pass


class TemporalT(Temporal):
    pass


class TemporalTau(Temporal):
    pass


def aztype(obj):
    for t in type(obj.azimuthal).__mro__:
        if t in (AzimuthalXY, AzimuthalRhoPhi):
            return t
    else:
        return None


def ltype(obj):
    for t in type(obj.longitudinal).__mro__:
        if t in (LongitudinalZ, LongitudinalTheta, LongitudinalEta):
            return t
    else:
        return None


def ttype(obj):
    for t in type(obj.temporal).__mro__:
        if t in (TemporalT, TemporalTau):
            return t
    else:
        return None


_coordinate_class_to_names = {
    AzimuthalXY: ("x", "y"),
    AzimuthalRhoPhi: ("rho", "phi"),
    LongitudinalZ: ("z",),
    LongitudinalTheta: ("theta",),
    LongitudinalEta: ("eta",),
    TemporalT: ("t",),
    TemporalTau: ("tau",),
}


_repr_generic_to_momentum = {
    "x": "px",
    "y": "py",
    "rho": "pt",
    "z": "pz",
    "t": "E",
    "tau": "mass",
}


_repr_momentum_to_generic = {
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


_coordinate_order = [
    "x",
    "px",
    "y",
    "py",
    "rho",
    "pt",
    "phi",
    "z",
    "pz",
    "theta",
    "eta",
    "t",
    "E",
    "e",
    "energy",
    "tau",
    "M",
    "m",
    "mass",
]
