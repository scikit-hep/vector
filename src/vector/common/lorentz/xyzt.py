# -*- coding: utf-8 -*-
# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import absolute_import, division, print_function

from typing import TYPE_CHECKING, TypeVar, overload

import numpy as np

import vector.core.lorentz.xyzt

if TYPE_CHECKING:
    from vector.protocols.lorentz import LorentzVector, Scalar


T = TypeVar("T")


class LorentzXYZTCommon:
    """
    This is the base class if you do not want magic methods.
    """

    @property
    def pt(self):
        # type: (LorentzVector) -> Scalar
        r"""
        The traverse momentum.

        Notes
        -----
        * ROOT name: Pt()

        .. math::

            \sqrt{\hat{x}^{2} + \hat{y}^{2}}
        """

        with np.errstate(invalid="ignore"):
            return vector.core.lorentz.xyzt.pt(self)

    @property
    def eta(self):
        # type: (LorentzVector) -> Scalar
        r"""
        The

        Notes
        -----
        * Other names: η
        * ROOT name: Eta()

        .. math::

            \arcsin{\frac{\hat{z}}{\sqrt{\hat{x}^{2} + \hat{y}^2}} }
        """

        with np.errstate(invalid="ignore"):
            return vector.core.lorentz.xyzt.eta(self)

    @property
    def phi(self):
        # type: (LorentzVector) -> Scalar
        r"""
        Notes
        -----
        * Other names: φ
        * ROOT name: Phi()

        .. math::

            \mathrm{arctan2}\left(\hat{y}, \hat{x}\right)
        """

        with np.errstate(invalid="ignore"):
            return vector.core.lorentz.xyzt.phi(self)

    @property
    def mass(self):
        # type: (LorentzVector) -> Scalar
        with np.errstate(invalid="ignore"):
            return vector.core.lorentz.xyzt.mag(self)

    @property
    def mag(self):
        # type: (LorentzVector) -> Scalar
        with np.errstate(invalid="ignore"):
            return vector.core.lorentz.xyzt.mag(self)

    @property
    def mag2(self):
        # type: (LorentzVector) -> Scalar
        with np.errstate(invalid="ignore"):
            return vector.core.lorentz.xyzt.mag2(self)

    def dot(self, other):
        # type: (LorentzVector, LorentzVector) -> Scalar
        return vector.core.lorentz.xyzt.dot(self, other)


class LorentzXYZTDunderMixin(LorentzXYZTCommon):
    """
    This is the base class with magic methods.
    """

    @overload
    def __add__(self, other):
        # type: (LorentzVector, LorentzVector) -> LorentzVector
        pass

    @overload
    def __add__(self, other):
        # type: (LorentzVector, Scalar) -> LorentzVector
        pass

    def __add__(self, other):
        if isinstance(other, LorentzXYZTCommon):
            return self.__class__(*vector.core.lorentz.xyzt.add(self, other))
        else:
            return self.__class__(*vector.core.lorentz.xyzt.add_scalar(self, other))

    @overload
    def __mul__(self, other):
        # type: (LorentzVector, LorentzVector) -> Scalar
        pass

    @overload
    def __mul__(self, other):
        # type: (LorentzVector, Scalar) -> LorentzVector
        pass

    def __mul__(self, other):
        if isinstance(other, LorentzXYZTCommon):
            return vector.core.lorentz.xyzt.dot(self, other)
        else:
            return self.__class__(
                *vector.core.lorentz.xyzt.multiply_scalar(self, other)
            )

    def __radd__(self, other):
        # type: (LorentzVector, Scalar) -> LorentzVector
        return self.__class__(*vector.core.lorentz.xyzt.add_scalar(self, other))

    def __rmul__(self, other):
        # type: (LorentzVector, Scalar) -> LorentzVector
        return self.__class__(*vector.core.lorentz.xyzt.multiply_scalar(self, other))
