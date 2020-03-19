# Copyright (c) 2019-2020, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import division, absolute_import, print_function

import numpy as np

from ...core import lorentz


class LorentzXYZCommon(object):
    @property
    def pt(self):
        r"""
        The traverse momentum.

        Notes
        -----
        * ROOT name: Pt()

        .. math::

            \sqrt{\hat{x}^{2} + \hat{y}^{2}}
        """

        with np.errstate(invalid="ignore"):
            return lorentz.xyzt.pt(self)

    @property
    def eta(self):
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
            return lorentz.xyzt.eta(self)

    @property
    def phi(self):
        r"""
        Notes
        -----
        * Other names: φ
        * ROOT name: Phi()

        .. math::

            \mathrm{arctan2}\left(\hat{y}, \hat{x}\right)
        """

        with np.errstate(invalid="ignore"):
            return lorentz.xyzt.phi(self)

    @property
    def mass(self):
        with np.errstate(invalid="ignore"):
            return lorentz.xyzt.mass(self)
