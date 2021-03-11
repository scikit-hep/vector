# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

import numpy

import vector.backends.object_
import vector.compute.lorentz
import vector.compute.planar
import vector.compute.spatial
import vector.geometry
import vector.methods
from vector.geometry import _coordinate_class_to_names


def _getitem(array, where):
    if isinstance(where, str):
        return array.view(numpy.ndarray)[where]
    else:
        out = numpy.ndarray.__getitem__(array, where)
        if isinstance(out, numpy.void):
            azimuthal, longitudinal, temporal = None, None, None
            if hasattr(array, "_azimuthal_type"):
                azimuthal = array._azimuthal_type.ObjectClass(
                    *[
                        out[x]
                        for x in _coordinate_class_to_names[
                            vector.geometry.aztype(array)
                        ]
                    ]
                )
            if hasattr(array, "_longitudinal_type"):
                longitudinal = array._longitudinal_type.ObjectClass(
                    *[
                        out[x]
                        for x in _coordinate_class_to_names[
                            vector.geometry.ltype(array)
                        ]
                    ]
                )
            if hasattr(array, "_temporal_type"):
                temporal = array._temporal_type.ObjectClass(
                    *[
                        out[x]
                        for x in _coordinate_class_to_names[
                            vector.geometry.ttype(array)
                        ]
                    ]
                )
            if temporal is not None:
                return array.ObjectClass(azimuthal, longitudinal, temporal)
            elif longitudinal is not None:
                return array.ObjectClass(azimuthal, longitudinal)
            elif azimuthal is not None:
                return array.ObjectClass(azimuthal)
            else:
                return array.ObjectClass(*out.view(numpy.ndarray))
        else:
            return out


def _array_repr(array):
    name = type(array).__name__
    return name + repr(array.view(numpy.ndarray))[5:].replace(
        "\n     ", "\n" + " " * len(name)
    )


def _has(array, names):
    dtype_names = array.dtype.names
    if dtype_names is None:
        dtype_names = ()
    return all(x in dtype_names for x in names)


class CoordinatesNumpy:
    lib = numpy


class AzimuthalNumpy(CoordinatesNumpy):
    pass


class LongitudinalNumpy(CoordinatesNumpy):
    pass


class TemporalNumpy(CoordinatesNumpy):
    pass


class VectorNumpy:
    pass


class AzimuthalNumpyXY(AzimuthalNumpy, vector.geometry.AzimuthalXY, numpy.ndarray):
    ObjectClass = vector.backends.object_.AzimuthalObjectXY

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("x", "y")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y")'
            )

    @property
    def elements(self):
        return (self["x"], self["y"])

    @property
    def x(self):
        return self["x"]

    @property
    def y(self):
        return self["y"]

    def __getitem__(self, where):
        return _getitem(self, where)


class AzimuthalNumpyRhoPhi(
    AzimuthalNumpy, vector.geometry.AzimuthalRhoPhi, numpy.ndarray
):
    ObjectClass = vector.backends.object_.AzimuthalObjectRhoPhi

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("rho", "phi")):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("rho", "phi")'
            )

    @property
    def elements(self):
        return (self["rho"], self["phi"])

    @property
    def rho(self):
        return self["rho"]

    @property
    def phi(self):
        return self["phi"]

    def __getitem__(self, where):
        return _getitem(self, where)


class LongitudinalNumpyZ(
    LongitudinalNumpy, vector.geometry.LongitudinalZ, numpy.ndarray
):
    ObjectClass = vector.backends.object_.LongitudinalObjectZ

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("z",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "z"'
            )

    @property
    def elements(self):
        return (self["z"],)

    @property
    def z(self):
        return self["z"]

    def __getitem__(self, where):
        return _getitem(self, where)


class LongitudinalNumpyTheta(
    LongitudinalNumpy, vector.geometry.LongitudinalTheta, numpy.ndarray
):
    ObjectClass = vector.backends.object_.LongitudinalObjectTheta

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("theta",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "theta"'
            )

    @property
    def elements(self):
        return (self["theta"],)

    @property
    def theta(self):
        return self["theta"]

    def __getitem__(self, where):
        return _getitem(self, where)


class LongitudinalNumpyEta(
    LongitudinalNumpy, vector.geometry.LongitudinalEta, numpy.ndarray
):
    ObjectClass = vector.backends.object_.LongitudinalObjectEta

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("eta",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "eta"'
            )

    @property
    def elements(self):
        return (self["eta"],)

    @property
    def eta(self):
        return self["eta"]

    def __getitem__(self, where):
        return _getitem(self, where)


class TemporalNumpyT(TemporalNumpy, vector.geometry.TemporalT, numpy.ndarray):
    ObjectClass = vector.backends.object_.TemporalObjectT

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("t",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "t"'
            )

    @property
    def elements(self):
        return (self["t"],)

    @property
    def t(self):
        return self["t"]

    def __getitem__(self, where):
        return _getitem(self, where)


class TemporalNumpyTau(TemporalNumpy, vector.geometry.TemporalTau, numpy.ndarray):
    ObjectClass = vector.backends.object_.TemporalObjectTau

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if not _has(self, ("tau",)):
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "tau"'
            )

    @property
    def elements(self):
        return (self["tau"],)

    @property
    def tau(self):
        return self["tau"]

    def __getitem__(self, where):
        return _getitem(self, where)


class VectorNumpy2D(
    VectorNumpy, vector.methods.Planar, vector.geometry.Vector2D, numpy.ndarray
):
    lib = numpy
    ObjectClass = vector.backends.object_.VectorObject2D

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if _has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif _has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi")'
            )

    def __str__(self):
        return str(self.view(numpy.ndarray))

    def __repr__(self):
        return _array_repr(self)

    def to_xy(self):
        return self._wrap_result(
            (
                vector.compute.planar.x.dispatch(self),
                vector.compute.planar.y.dispatch(self),
            ),
            [vector.geometry.AzimuthalXY],
        )

    def to_rhophi(self):
        return self._wrap_result(
            (
                vector.compute.planar.rho.dispatch(self),
                vector.compute.planar.phi.dispatch(self),
            ),
            [vector.geometry.AzimuthalRhoPhi],
        )

    @property
    def azimuthal(self):
        return self.view(self._azimuthal_type)

    def _wrap_result(self, result, returns):
        if returns == [float]:
            return result

        elif returns == [vector.geometry.AzimuthalXY] or returns == [
            vector.geometry.AzimuthalRhoPhi
        ]:
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            out = numpy.empty(result[0].shape, dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            return out.view(type(self))

        else:
            raise AssertionError(repr(returns))

    def __getitem__(self, where):
        return _getitem(self, where)


class MomentumNumpy2D(vector.methods.PlanarMomentum, VectorNumpy2D):
    ObjectClass = vector.backends.object_.MomentumObject2D


class VectorNumpy3D(
    VectorNumpy, vector.methods.Spatial, vector.geometry.Vector3D, numpy.ndarray
):
    lib = numpy
    ObjectClass = vector.backends.object_.VectorObject3D
    ProjectionClass2D = VectorNumpy2D

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if _has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif _has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi")'
            )
        if _has(self, ("z",)):
            self._longitudinal_type = LongitudinalNumpyZ
        elif _has(self, ("theta",)):
            self._longitudinal_type = LongitudinalNumpyTheta
        elif _has(self, ("eta",)):
            self._longitudinal_type = LongitudinalNumpyEta
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "z" or "theta" or "eta"'
            )

    def __str__(self):
        return str(self.view(numpy.ndarray))

    def __repr__(self):
        return _array_repr(self)

    def to_xyz(self):
        return self._wrap_result(
            (
                vector.compute.planar.x.dispatch(self),
                vector.compute.planar.y.dispatch(self),
                vector.compute.spatial.z.dispatch(self),
            ),
            [vector.geometry.AzimuthalXY, vector.geometry.LongitudinalZ],
        )

    def to_xytheta(self):
        return self._wrap_result(
            (
                vector.compute.planar.x.dispatch(self),
                vector.compute.planar.y.dispatch(self),
                vector.compute.spatial.theta.dispatch(self),
            ),
            [vector.geometry.AzimuthalXY, vector.geometry.LongitudinalTheta],
        )

    def to_xyeta(self):
        return self._wrap_result(
            (
                vector.compute.planar.x.dispatch(self),
                vector.compute.planar.y.dispatch(self),
                vector.compute.spatial.eta.dispatch(self),
            ),
            [vector.geometry.AzimuthalXY, vector.geometry.LongitudinalEta],
        )

    def to_rhophiz(self):
        return self._wrap_result(
            (
                vector.compute.planar.rho.dispatch(self),
                vector.compute.planar.phi.dispatch(self),
                vector.compute.spatial.z.dispatch(self),
            ),
            [vector.geometry.AzimuthalRhoPhi, vector.geometry.LongitudinalZ],
        )

    def to_rhophitheta(self):
        return self._wrap_result(
            (
                vector.compute.planar.rho.dispatch(self),
                vector.compute.planar.phi.dispatch(self),
                vector.compute.spatial.theta.dispatch(self),
            ),
            [vector.geometry.AzimuthalRhoPhi, vector.geometry.LongitudinalTheta],
        )

    def to_rhophieta(self):
        return self._wrap_result(
            (
                vector.compute.planar.rho.dispatch(self),
                vector.compute.planar.phi.dispatch(self),
                vector.compute.spatial.eta.dispatch(self),
            ),
            [vector.geometry.AzimuthalRhoPhi, vector.geometry.LongitudinalEta],
        )

    @property
    def azimuthal(self):
        return self.view(self._azimuthal_type)

    @property
    def longitudinal(self):
        return self.view(self._longitudinal_type)

    def __getitem__(self, where):
        return _getitem(self, where)

    def _wrap_result(self, result, returns):
        if returns == [float]:
            return result

        elif returns == [vector.geometry.AzimuthalXY] or returns == [
            vector.geometry.AzimuthalRhoPhi
        ]:
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[vector.geometry.ltype(self)]:
                dtype.append((name, self.dtype[name]))
            out = numpy.empty(result[0].shape, dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[vector.geometry.ltype(self)]:
                out[name] = self[name]
            return out.view(type(self))

        elif (
            (len(returns) == 2 or (len(returns) == 3 and returns[2] is None))
            and isinstance(returns[0], type)
            and issubclass(returns[0], vector.geometry.Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], vector.geometry.Longitudinal)
        ):
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                dtype.append((name, result[i].dtype))
                i += 1
            out = numpy.empty(result[0].shape, dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                out[name] = result[i]
                i += 1
            return out.view(type(self))

        else:
            raise AssertionError(repr(returns))


class MomentumNumpy3D(vector.methods.SpatialMomentum, VectorNumpy3D):
    ObjectClass = vector.backends.object_.MomentumObject3D
    ProjectionClass2D = MomentumNumpy2D


class VectorNumpy4D(
    VectorNumpy, vector.methods.Lorentz, vector.geometry.Vector4D, numpy.ndarray
):
    lib = numpy
    ObjectClass = vector.backends.object_.VectorObject4D
    ProjectionClass2D = VectorNumpy2D
    ProjectionClass3D = VectorNumpy3D

    def __new__(cls, *args, **kwargs):
        return numpy.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        if _has(self, ("x", "y")):
            self._azimuthal_type = AzimuthalNumpyXY
        elif _has(self, ("rho", "phi")):
            self._azimuthal_type = AzimuthalNumpyRhoPhi
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'fields ("x", "y") or ("rho", "phi")'
            )
        if _has(self, ("z",)):
            self._longitudinal_type = LongitudinalNumpyZ
        elif _has(self, ("theta",)):
            self._longitudinal_type = LongitudinalNumpyTheta
        elif _has(self, ("eta",)):
            self._longitudinal_type = LongitudinalNumpyEta
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "z" or "theta" or "eta"'
            )
        if _has(self, ("t",)):
            self._temporal_type = TemporalNumpyT
        elif _has(self, ("tau",)):
            self._temporal_type = TemporalNumpyTau
        else:
            raise TypeError(
                f"{type(self).__name__} must have a structured dtype containing "
                'field "t" or "tau"'
            )

    def __str__(self):
        return str(self.view(numpy.ndarray))

    def __repr__(self):
        return _array_repr(self)

    def to_xyzt(self):
        return self._wrap_result(
            (
                vector.compute.planar.x.dispatch(self),
                vector.compute.planar.y.dispatch(self),
                vector.compute.spatial.z.dispatch(self),
                vector.compute.lorentz.t.dispatch(self),
            ),
            [
                vector.geometry.AzimuthalXY,
                vector.geometry.LongitudinalZ,
                vector.geometry.TemporalT,
            ],
        )

    def to_xyztau(self):
        return self._wrap_result(
            (
                vector.compute.planar.x.dispatch(self),
                vector.compute.planar.y.dispatch(self),
                vector.compute.spatial.z.dispatch(self),
                vector.compute.lorentz.tau.dispatch(self),
            ),
            [
                vector.geometry.AzimuthalXY,
                vector.geometry.LongitudinalZ,
                vector.geometry.TemporalTau,
            ],
        )

    def to_xythetat(self):
        return self._wrap_result(
            (
                vector.compute.planar.x.dispatch(self),
                vector.compute.planar.y.dispatch(self),
                vector.compute.spatial.theta.dispatch(self),
                vector.compute.lorentz.t.dispatch(self),
            ),
            [
                vector.geometry.AzimuthalXY,
                vector.geometry.LongitudinalTheta,
                vector.geometry.TemporalT,
            ],
        )

    def to_xythetatau(self):
        return self._wrap_result(
            (
                vector.compute.planar.x.dispatch(self),
                vector.compute.planar.y.dispatch(self),
                vector.compute.spatial.theta.dispatch(self),
                vector.compute.lorentz.tau.dispatch(self),
            ),
            [
                vector.geometry.AzimuthalXY,
                vector.geometry.LongitudinalTheta,
                vector.geometry.TemporalTau,
            ],
        )

    def to_xyetat(self):
        return self._wrap_result(
            (
                vector.compute.planar.x.dispatch(self),
                vector.compute.planar.y.dispatch(self),
                vector.compute.spatial.eta.dispatch(self),
                vector.compute.lorentz.t.dispatch(self),
            ),
            [
                vector.geometry.AzimuthalXY,
                vector.geometry.LongitudinalEta,
                vector.geometry.TemporalT,
            ],
        )

    def to_xyetatau(self):
        return self._wrap_result(
            (
                vector.compute.planar.x.dispatch(self),
                vector.compute.planar.y.dispatch(self),
                vector.compute.spatial.eta.dispatch(self),
                vector.compute.lorentz.tau.dispatch(self),
            ),
            [
                vector.geometry.AzimuthalXY,
                vector.geometry.LongitudinalEta,
                vector.geometry.TemporalTau,
            ],
        )

    def to_rhophizt(self):
        return self._wrap_result(
            (
                vector.compute.planar.rho.dispatch(self),
                vector.compute.planar.phi.dispatch(self),
                vector.compute.spatial.z.dispatch(self),
                vector.compute.lorentz.t.dispatch(self),
            ),
            [
                vector.geometry.AzimuthalRhoPhi,
                vector.geometry.LongitudinalZ,
                vector.geometry.TemporalT,
            ],
        )

    def to_rhophiztau(self):
        return self._wrap_result(
            (
                vector.compute.planar.rho.dispatch(self),
                vector.compute.planar.phi.dispatch(self),
                vector.compute.spatial.z.dispatch(self),
                vector.compute.lorentz.tau.dispatch(self),
            ),
            [
                vector.geometry.AzimuthalRhoPhi,
                vector.geometry.LongitudinalZ,
                vector.geometry.TemporalTau,
            ],
        )

    def to_rhophithetat(self):
        return self._wrap_result(
            (
                vector.compute.planar.rho.dispatch(self),
                vector.compute.planar.phi.dispatch(self),
                vector.compute.spatial.theta.dispatch(self),
                vector.compute.lorentz.t.dispatch(self),
            ),
            [
                vector.geometry.AzimuthalRhoPhi,
                vector.geometry.LongitudinalTheta,
                vector.geometry.TemporalT,
            ],
        )

    def to_rhophithetatau(self):
        return self._wrap_result(
            (
                vector.compute.planar.rho.dispatch(self),
                vector.compute.planar.phi.dispatch(self),
                vector.compute.spatial.theta.dispatch(self),
                vector.compute.lorentz.tau.dispatch(self),
            ),
            [
                vector.geometry.AzimuthalRhoPhi,
                vector.geometry.LongitudinalTheta,
                vector.geometry.TemporalTau,
            ],
        )

    def to_rhophietat(self):
        return self._wrap_result(
            (
                vector.compute.planar.rho.dispatch(self),
                vector.compute.planar.phi.dispatch(self),
                vector.compute.spatial.eta.dispatch(self),
                vector.compute.lorentz.t.dispatch(self),
            ),
            [
                vector.geometry.AzimuthalRhoPhi,
                vector.geometry.LongitudinalEta,
                vector.geometry.TemporalT,
            ],
        )

    def to_rhophietatau(self):
        return self._wrap_result(
            (
                vector.compute.planar.rho.dispatch(self),
                vector.compute.planar.phi.dispatch(self),
                vector.compute.spatial.eta.dispatch(self),
                vector.compute.lorentz.tau.dispatch(self),
            ),
            [
                vector.geometry.AzimuthalRhoPhi,
                vector.geometry.LongitudinalEta,
                vector.geometry.TemporalTau,
            ],
        )

    @property
    def azimuthal(self):
        return self.view(self._azimuthal_type)

    @property
    def longitudinal(self):
        return self.view(self._longitudinal_type)

    @property
    def temporal(self):
        return self.view(self._temporal_type)

    def __getitem__(self, where):
        return _getitem(self, where)

    def _wrap_result(self, result, returns):
        if returns == [float]:
            return result

        elif returns == [vector.geometry.AzimuthalXY] or returns == [
            vector.geometry.AzimuthalRhoPhi
        ]:
            dtype = []
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                dtype.append((name, result[i].dtype))
            for name in _coordinate_class_to_names[vector.geometry.ltype(self)]:
                dtype.append((name, self.dtype[name]))
            for name in _coordinate_class_to_names[vector.geometry.ttype(self)]:
                dtype.append((name, self.dtype[name]))
            out = numpy.empty(result[0].shape, dtype=dtype)
            for i, name in enumerate(_coordinate_class_to_names[returns[0]]):
                out[name] = result[i]
            for name in _coordinate_class_to_names[vector.geometry.ltype(self)]:
                out[name] = self[name]
            for name in _coordinate_class_to_names[vector.geometry.ttype(self)]:
                out[name] = self[name]
            return out.view(type(self))

        elif (
            len(returns) == 2
            and isinstance(returns[0], type)
            and issubclass(returns[0], vector.geometry.Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], vector.geometry.Longitudinal)
        ):
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[vector.geometry.ttype(self)]:
                dtype.append((name, self.dtype[name]))
            out = numpy.empty(result[0].shape, dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[vector.geometry.ttype(self)]:
                out[name] = self[name]
            return out.view(type(self))

        elif (
            len(returns) == 3
            and isinstance(returns[0], type)
            and issubclass(returns[0], vector.geometry.Azimuthal)
            and isinstance(returns[1], type)
            and issubclass(returns[1], vector.geometry.Longitudinal)
        ):
            dtype = []
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                dtype.append((name, result[i].dtype))
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                dtype.append((name, result[i].dtype))
                i += 1
            is_4d = False
            if isinstance(returns[2], type) and issubclass(
                returns[2], vector.geometry.Temporal
            ):
                is_4d = True
                for name in _coordinate_class_to_names[returns[2]]:
                    dtype.append((name, result[i].dtype))
                    i += 1
            elif returns[2] is not None:
                raise AssertionError(repr(type(returns[2])))
            out = numpy.empty(result[0].shape, dtype=dtype)
            i = 0
            for name in _coordinate_class_to_names[returns[0]]:
                out[name] = result[i]
                i += 1
            for name in _coordinate_class_to_names[returns[1]]:
                out[name] = result[i]
                i += 1
            if is_4d:
                for name in _coordinate_class_to_names[returns[2]]:
                    out[name] = result[i]
                    i += 1
                return out.view(type(self))
            else:
                return out.view(self.ProjectionClass3D)

        else:
            raise AssertionError(repr(returns))


class MomentumNumpy4D(vector.methods.LorentzMomentum, VectorNumpy4D):
    ObjectClass = vector.backends.object_.MomentumObject4D
    ProjectionClass2D = MomentumNumpy2D
    ProjectionClass3D = MomentumNumpy3D
