# -*- coding: utf-8 -*-
# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import absolute_import, division, print_function

from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from vector.protocols.lorentz import LorentzVector, Scalar

    T = TypeVar("T", bound="LorentzVector")

import vector.core.lorentz.xyzt


class LorentzXYZTCommon(object):
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

    def mul(self, other):
        # type: (T, Scalar) -> T
        return self.__class__(*vector.core.lorentz.xyzt.multiply_scalar(self, other))

    def dot(self, other):
        # type: (T, T) -> Scalar
        return vector.core.lorentz.xyzt.dot(self, other)


class LorentzXYZTNormal(LorentzXYZTCommon):
    """
    This is the base class with magic methods.
    """

    def __add__(self, other):
        # type: (T, LorentzVector) -> T
        return self.__class__(*vector.core.lorentz.xyzt.add(self, other))

    def __mul__(self, other):
        return (
            self.dot(other) if isinstance(other, LorentzXYZTCommon) else self.mul(other)
        )
