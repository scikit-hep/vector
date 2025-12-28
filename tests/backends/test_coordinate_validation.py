# Copyright (c) 2019-2025, Saransh Chopra, Henry Schreiner, Eduardo Rodrigues, Jonas Eschle, and Jim Pivarski.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

from __future__ import annotations

import numpy as np
import pytest

import vector

ak = pytest.importorskip("awkward")

pytestmark = pytest.mark.awkward


# ============================================================================
# Duplicate temporal coordinates (t-like vs tau-like)
# ============================================================================
# Temporal coordinates: t, E, e, energy (all map to 't')
#                       tau, M, m, mass (all map to 'tau')
# These are mutually exclusive


def test_duplicate_E_e_object():
    """vector.obj should reject E + e"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(pt=1.0, phi=0.5, eta=1.0, E=5.0, e=5.0)


def test_duplicate_E_e_numpy():
    """vector.array should reject E + e"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.array(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "E": np.array([5.0, 6.0]),
                "e": np.array([5.0, 6.0]),
            }
        )


def test_duplicate_E_e_awkward():
    """vector.Array should reject E + e"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.Array(
            ak.Array(
                {
                    "pt": np.array([1.0, 2.0]),
                    "phi": np.array([0.5, 1.0]),
                    "eta": np.array([1.0, 1.5]),
                    "E": np.array([5.0, 6.0]),
                    "e": np.array([5.0, 6.0]),
                }
            )
        )


def test_duplicate_E_e_zip():
    """vector.zip should reject E + e"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.zip(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "E": np.array([5.0, 6.0]),
                "e": np.array([5.0, 6.0]),
            }
        )


def test_duplicate_E_energy_object():
    """vector.obj should reject E + energy"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(pt=1.0, phi=0.5, eta=1.0, E=5.0, energy=5.0)


def test_duplicate_E_energy_numpy():
    """vector.array should reject E + energy"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.array(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "E": np.array([5.0, 6.0]),
                "energy": np.array([5.0, 6.0]),
            }
        )


def test_duplicate_E_energy_awkward():
    """vector.Array should reject E + energy"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.Array(
            ak.Array(
                {
                    "pt": np.array([1.0, 2.0]),
                    "phi": np.array([0.5, 1.0]),
                    "eta": np.array([1.0, 1.5]),
                    "E": np.array([5.0, 6.0]),
                    "energy": np.array([5.0, 6.0]),
                }
            )
        )


def test_duplicate_E_energy_zip():
    """vector.zip should reject E + energy"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.zip(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "E": np.array([5.0, 6.0]),
                "energy": np.array([5.0, 6.0]),
            }
        )


def test_duplicate_e_energy_object():
    """vector.obj should reject e + energy"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(pt=1.0, phi=0.5, eta=1.0, e=5.0, energy=5.0)


def test_duplicate_e_energy_numpy():
    """vector.array should reject e + energy"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.array(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "e": np.array([5.0, 6.0]),
                "energy": np.array([5.0, 6.0]),
            }
        )


def test_duplicate_e_energy_awkward():
    """vector.Array should reject e + energy"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.Array(
            ak.Array(
                {
                    "pt": np.array([1.0, 2.0]),
                    "phi": np.array([0.5, 1.0]),
                    "eta": np.array([1.0, 1.5]),
                    "e": np.array([5.0, 6.0]),
                    "energy": np.array([5.0, 6.0]),
                }
            )
        )


def test_duplicate_e_energy_zip():
    """vector.zip should reject e + energy"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.zip(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "e": np.array([5.0, 6.0]),
                "energy": np.array([5.0, 6.0]),
            }
        )


def test_duplicate_M_m_object():
    """vector.obj should reject M + m"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(pt=1.0, phi=0.5, eta=1.0, M=0.5, m=0.5)


def test_duplicate_M_m_numpy():
    """vector.array should reject M + m"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.array(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "M": np.array([0.5, 0.5]),
                "m": np.array([0.5, 0.5]),
            }
        )


def test_duplicate_M_m_awkward():
    """vector.Array should reject M + m"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.Array(
            ak.Array(
                {
                    "pt": np.array([1.0, 2.0]),
                    "phi": np.array([0.5, 1.0]),
                    "eta": np.array([1.0, 1.5]),
                    "M": np.array([0.5, 0.5]),
                    "m": np.array([0.5, 0.5]),
                }
            )
        )


def test_duplicate_M_m_zip():
    """vector.zip should reject M + m"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.zip(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "M": np.array([0.5, 0.5]),
                "m": np.array([0.5, 0.5]),
            }
        )


def test_duplicate_M_mass_object():
    """vector.obj should reject M + mass"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(pt=1.0, phi=0.5, eta=1.0, M=0.5, mass=0.5)


def test_duplicate_M_mass_numpy():
    """vector.array should reject M + mass"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.array(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "M": np.array([0.5, 0.5]),
                "mass": np.array([0.5, 0.5]),
            }
        )


def test_duplicate_M_mass_awkward():
    """vector.Array should reject M + mass"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.Array(
            ak.Array(
                {
                    "pt": np.array([1.0, 2.0]),
                    "phi": np.array([0.5, 1.0]),
                    "eta": np.array([1.0, 1.5]),
                    "M": np.array([0.5, 0.5]),
                    "mass": np.array([0.5, 0.5]),
                }
            )
        )


def test_duplicate_M_mass_zip():
    """vector.zip should reject M + mass"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.zip(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "M": np.array([0.5, 0.5]),
                "mass": np.array([0.5, 0.5]),
            }
        )


def test_duplicate_m_mass_object():
    """vector.obj should reject m + mass"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(pt=1.0, phi=0.5, eta=1.0, m=0.5, mass=0.5)


def test_duplicate_m_mass_numpy():
    """vector.array should reject m + mass"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.array(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "m": np.array([0.5, 0.5]),
                "mass": np.array([0.5, 0.5]),
            }
        )


def test_duplicate_m_mass_awkward():
    """vector.Array should reject m + mass"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.Array(
            ak.Array(
                {
                    "pt": np.array([1.0, 2.0]),
                    "phi": np.array([0.5, 1.0]),
                    "eta": np.array([1.0, 1.5]),
                    "m": np.array([0.5, 0.5]),
                    "mass": np.array([0.5, 0.5]),
                }
            )
        )


def test_duplicate_m_mass_zip():
    """vector.zip should reject m + mass"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.zip(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "m": np.array([0.5, 0.5]),
                "mass": np.array([0.5, 0.5]),
            }
        )


def test_conflicting_energy_mass_object():
    """vector.obj should reject energy + mass (t-like + tau-like)"""
    with pytest.raises(TypeError, match="specify t= or tau=, but not more than one"):
        vector.obj(pt=1.0, phi=0.5, eta=1.0, energy=5.0, mass=0.5)


def test_conflicting_energy_mass_numpy():
    """vector.array should reject energy + mass (t-like + tau-like)"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.array(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "energy": np.array([5.0, 6.0]),
                "mass": np.array([0.5, 0.5]),
            }
        )


def test_conflicting_energy_mass_awkward():
    """vector.Array should reject energy + mass (t-like + tau-like)"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.Array(
            ak.Array(
                {
                    "pt": np.array([1.0, 2.0]),
                    "phi": np.array([0.5, 1.0]),
                    "eta": np.array([1.0, 1.5]),
                    "energy": np.array([5.0, 6.0]),
                    "mass": np.array([0.5, 0.5]),
                }
            )
        )


def test_conflicting_energy_mass_zip():
    """vector.zip should reject energy + mass (t-like + tau-like)"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.zip(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "energy": np.array([5.0, 6.0]),
                "mass": np.array([0.5, 0.5]),
            }
        )


def test_conflicting_t_tau_object():
    """vector.obj should reject t + tau"""
    with pytest.raises(TypeError, match="specify t= or tau="):
        vector.obj(x=1.0, y=2.0, z=3.0, t=5.0, tau=0.5)


def test_conflicting_t_tau_numpy():
    """vector.array should reject t + tau"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.array(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([2.0, 3.0]),
                "z": np.array([3.0, 4.0]),
                "t": np.array([5.0, 6.0]),
                "tau": np.array([0.5, 0.5]),
            }
        )


def test_conflicting_t_tau_awkward():
    """vector.Array should reject t + tau"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.Array(
            ak.Array(
                {
                    "x": np.array([1.0, 2.0]),
                    "y": np.array([2.0, 3.0]),
                    "z": np.array([3.0, 4.0]),
                    "t": np.array([5.0, 6.0]),
                    "tau": np.array([0.5, 0.5]),
                }
            )
        )


def test_conflicting_t_tau_zip():
    """vector.zip should reject t + tau"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.zip(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([2.0, 3.0]),
                "z": np.array([3.0, 4.0]),
                "t": np.array([5.0, 6.0]),
                "tau": np.array([0.5, 0.5]),
            }
        )


def test_conflicting_E_mass_object():
    """vector.obj should reject E + mass"""
    with pytest.raises(TypeError, match="specify t= or tau=, but not more than one"):
        vector.obj(pt=1.0, phi=0.5, eta=1.0, E=5.0, mass=0.5)


def test_conflicting_E_mass_numpy():
    """vector.array should reject E + mass"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.array(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "E": np.array([5.0, 6.0]),
                "mass": np.array([0.5, 0.5]),
            }
        )


def test_conflicting_E_mass_awkward():
    """vector.Array should reject E + mass"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.Array(
            ak.Array(
                {
                    "pt": np.array([1.0, 2.0]),
                    "phi": np.array([0.5, 1.0]),
                    "eta": np.array([1.0, 1.5]),
                    "E": np.array([5.0, 6.0]),
                    "mass": np.array([0.5, 0.5]),
                }
            )
        )


def test_conflicting_E_mass_zip():
    """vector.zip should reject E + mass"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.zip(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "E": np.array([5.0, 6.0]),
                "mass": np.array([0.5, 0.5]),
            }
        )


def test_conflicting_t_mass_object():
    """vector.obj should reject t + mass"""
    with pytest.raises(TypeError, match="specify t= or tau=, but not more than one"):
        vector.obj(x=1.0, y=2.0, z=3.0, t=5.0, mass=0.5)


def test_conflicting_t_mass_numpy():
    """vector.array should reject t + mass"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.array(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([2.0, 3.0]),
                "z": np.array([3.0, 4.0]),
                "t": np.array([5.0, 6.0]),
                "mass": np.array([0.5, 0.5]),
            }
        )


def test_conflicting_t_mass_awkward():
    """vector.Array should reject t + mass"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.Array(
            ak.Array(
                {
                    "x": np.array([1.0, 2.0]),
                    "y": np.array([2.0, 3.0]),
                    "z": np.array([3.0, 4.0]),
                    "t": np.array([5.0, 6.0]),
                    "mass": np.array([0.5, 0.5]),
                }
            )
        )


def test_conflicting_t_mass_zip():
    """vector.zip should reject t + mass"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.zip(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([2.0, 3.0]),
                "z": np.array([3.0, 4.0]),
                "t": np.array([5.0, 6.0]),
                "mass": np.array([0.5, 0.5]),
            }
        )


def test_conflicting_energy_tau_object():
    """vector.obj should reject energy + tau"""
    with pytest.raises(TypeError, match="specify t= or tau=, but not more than one"):
        vector.obj(x=1.0, y=2.0, z=3.0, energy=5.0, tau=0.5)


def test_conflicting_energy_tau_numpy():
    """vector.array should reject energy + tau"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.array(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([2.0, 3.0]),
                "z": np.array([3.0, 4.0]),
                "energy": np.array([5.0, 6.0]),
                "tau": np.array([0.5, 0.5]),
            }
        )


def test_conflicting_energy_tau_awkward():
    """vector.Array should reject energy + tau"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.Array(
            ak.Array(
                {
                    "x": np.array([1.0, 2.0]),
                    "y": np.array([2.0, 3.0]),
                    "z": np.array([3.0, 4.0]),
                    "energy": np.array([5.0, 6.0]),
                    "tau": np.array([0.5, 0.5]),
                }
            )
        )


def test_conflicting_energy_tau_zip():
    """vector.zip should reject energy + tau"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.zip(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([2.0, 3.0]),
                "z": np.array([3.0, 4.0]),
                "energy": np.array([5.0, 6.0]),
                "tau": np.array([0.5, 0.5]),
            }
        )


# ============================================================================
# Duplicate azimuthal coordinates
# ============================================================================
# x <-> px, y <-> py, rho <-> pt


def test_duplicate_px_x_object():
    """vector.obj should reject px + x"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.obj(px=1.0, x=1.0, y=2.0, z=3.0, t=5.0)


def test_duplicate_px_x_numpy():
    """vector.array should reject px + x"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.array(
            {
                "px": np.array([1.0, 2.0]),
                "x": np.array([1.0, 2.0]),
                "y": np.array([2.0, 3.0]),
                "z": np.array([3.0, 4.0]),
                "t": np.array([5.0, 6.0]),
            }
        )


def test_duplicate_px_x_awkward():
    """vector.Array should reject px + x"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.Array(
            ak.Array(
                {
                    "px": np.array([1.0, 2.0]),
                    "x": np.array([1.0, 2.0]),
                    "y": np.array([2.0, 3.0]),
                    "z": np.array([3.0, 4.0]),
                    "t": np.array([5.0, 6.0]),
                }
            )
        )


def test_duplicate_px_x_zip():
    """vector.zip should reject px + x"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.zip(
            {
                "px": np.array([1.0, 2.0]),
                "x": np.array([1.0, 2.0]),
                "y": np.array([2.0, 3.0]),
                "z": np.array([3.0, 4.0]),
                "t": np.array([5.0, 6.0]),
            }
        )


def test_duplicate_py_y_object():
    """vector.obj should reject py + y"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.obj(x=1.0, py=2.0, y=2.0, z=3.0, t=5.0)


def test_duplicate_py_y_numpy():
    """vector.array should reject py + y"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.array(
            {
                "x": np.array([1.0, 2.0]),
                "py": np.array([2.0, 3.0]),
                "y": np.array([2.0, 3.0]),
                "z": np.array([3.0, 4.0]),
                "t": np.array([5.0, 6.0]),
            }
        )


def test_duplicate_py_y_awkward():
    """vector.Array should reject py + y"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.Array(
            ak.Array(
                {
                    "x": np.array([1.0, 2.0]),
                    "py": np.array([2.0, 3.0]),
                    "y": np.array([2.0, 3.0]),
                    "z": np.array([3.0, 4.0]),
                    "t": np.array([5.0, 6.0]),
                }
            )
        )


def test_duplicate_py_y_zip():
    """vector.zip should reject py + y"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.zip(
            {
                "x": np.array([1.0, 2.0]),
                "py": np.array([2.0, 3.0]),
                "y": np.array([2.0, 3.0]),
                "z": np.array([3.0, 4.0]),
                "t": np.array([5.0, 6.0]),
            }
        )


def test_duplicate_pt_rho_object():
    """vector.obj should reject pt + rho"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.obj(pt=1.0, rho=1.0, phi=0.5, eta=1.0, mass=0.5)


def test_duplicate_pt_rho_numpy():
    """vector.array should reject pt + rho"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.array(
            {
                "pt": np.array([1.0, 2.0]),
                "rho": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "mass": np.array([0.5, 0.5]),
            }
        )


def test_duplicate_pt_rho_awkward():
    """vector.Array should reject pt + rho"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.Array(
            ak.Array(
                {
                    "pt": np.array([1.0, 2.0]),
                    "rho": np.array([1.0, 2.0]),
                    "phi": np.array([0.5, 1.0]),
                    "eta": np.array([1.0, 1.5]),
                    "mass": np.array([0.5, 0.5]),
                }
            )
        )


def test_duplicate_pt_rho_zip():
    """vector.zip should reject pt + rho"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.zip(
            {
                "pt": np.array([1.0, 2.0]),
                "rho": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "mass": np.array([0.5, 0.5]),
            }
        )


# ============================================================================
# Duplicate longitudinal coordinates
# ============================================================================
# z <-> pz


def test_duplicate_pz_z_object():
    """vector.obj should reject pz + z"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.obj(x=1.0, y=2.0, pz=3.0, z=3.0, t=5.0)


def test_duplicate_pz_z_numpy():
    """vector.array should reject pz + z"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.array(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([2.0, 3.0]),
                "pz": np.array([3.0, 4.0]),
                "z": np.array([3.0, 4.0]),
                "t": np.array([5.0, 6.0]),
            }
        )


def test_duplicate_pz_z_awkward():
    """vector.Array should reject pz + z"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.Array(
            ak.Array(
                {
                    "x": np.array([1.0, 2.0]),
                    "y": np.array([2.0, 3.0]),
                    "pz": np.array([3.0, 4.0]),
                    "z": np.array([3.0, 4.0]),
                    "t": np.array([5.0, 6.0]),
                }
            )
        )


def test_duplicate_pz_z_zip():
    """vector.zip should reject pz + z"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.zip(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([2.0, 3.0]),
                "pz": np.array([3.0, 4.0]),
                "z": np.array([3.0, 4.0]),
                "t": np.array([5.0, 6.0]),
            }
        )


# ============================================================================
# Mixed azimuthal coordinate systems (from _gather_coordinates)
# ============================================================================


def test_mixed_xy_with_rho_object():
    """vector.obj should reject x+y with rho"""
    with pytest.raises(TypeError, match="specify x= and y= or rho= and phi="):
        vector.obj(x=1.0, y=2.0, rho=1.0, z=3.0)


def test_mixed_xy_with_phi_object():
    """vector.obj should reject x+y with phi"""
    with pytest.raises(TypeError, match="specify x= and y= or rho= and phi="):
        vector.obj(x=1.0, y=2.0, phi=0.5, z=3.0)


def test_mixed_rhophi_with_x_object():
    """vector.obj should reject rho+phi with x"""
    with pytest.raises(TypeError, match="specify x= and y= or rho= and phi="):
        vector.obj(rho=1.0, phi=0.5, x=1.0, z=3.0)


def test_mixed_rhophi_with_y_object():
    """vector.obj should reject rho+phi with y"""
    with pytest.raises(TypeError, match="specify x= and y= or rho= and phi="):
        vector.obj(rho=1.0, phi=0.5, y=2.0, z=3.0)


# ============================================================================
# Mixed longitudinal coordinates (from _gather_coordinates)
# ============================================================================


def test_mixed_z_theta_object():
    """vector.obj should reject z with theta"""
    with pytest.raises(TypeError, match="specify z= or theta= or eta="):
        vector.obj(x=1.0, y=2.0, z=3.0, theta=1.0)


def test_mixed_z_eta_object():
    """vector.obj should reject z with eta"""
    with pytest.raises(TypeError, match="specify z= or theta= or eta="):
        vector.obj(x=1.0, y=2.0, z=3.0, eta=1.0)


def test_mixed_theta_eta_object():
    """vector.obj should reject theta with eta"""
    with pytest.raises(TypeError, match="specify z= or theta= or eta="):
        vector.obj(x=1.0, y=2.0, theta=1.0, eta=1.0)


# ============================================================================
# Valid combinations (ensure validation doesn't reject valid inputs)
# ============================================================================


def test_valid_pt_phi_eta_mass_object():
    """vector.obj should accept pt, phi, eta, mass"""
    vec = vector.obj(pt=1.0, phi=0.5, eta=1.0, mass=0.5)
    assert vec.pt == 1.0
    assert vec.phi == 0.5
    assert vec.eta == 1.0
    assert vec.mass == 0.5


def test_valid_pt_phi_eta_mass_numpy():
    """vector.array should accept pt, phi, eta, mass"""
    arr = vector.array(
        {
            "pt": np.array([1.0, 2.0]),
            "phi": np.array([0.5, 1.0]),
            "eta": np.array([1.0, 1.5]),
            "mass": np.array([0.5, 0.5]),
        }
    )
    assert np.allclose(arr.pt, [1.0, 2.0])
    assert np.allclose(arr.phi, [0.5, 1.0])
    assert np.allclose(arr.eta, [1.0, 1.5])
    assert np.allclose(arr.mass, [0.5, 0.5])


def test_valid_pt_phi_eta_mass_awkward():
    """vector.Array should accept pt, phi, eta, mass"""
    arr = vector.Array(
        ak.Array(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "eta": np.array([1.0, 1.5]),
                "mass": np.array([0.5, 0.5]),
            }
        )
    )
    assert ak.all(arr.pt == ak.Array([1.0, 2.0]))
    assert ak.all(arr.phi == ak.Array([0.5, 1.0]))
    assert ak.all(arr.eta == ak.Array([1.0, 1.5]))
    assert ak.all(arr.mass == ak.Array([0.5, 0.5]))


def test_valid_pt_phi_eta_mass_zip():
    """vector.zip should accept pt, phi, eta, mass"""
    arr = vector.zip(
        {
            "pt": np.array([1.0, 2.0]),
            "phi": np.array([0.5, 1.0]),
            "eta": np.array([1.0, 1.5]),
            "mass": np.array([0.5, 0.5]),
        }
    )
    assert ak.all(arr.pt == ak.Array([1.0, 2.0]))
    assert ak.all(arr.phi == ak.Array([0.5, 1.0]))
    assert ak.all(arr.eta == ak.Array([1.0, 1.5]))
    assert ak.all(arr.mass == ak.Array([0.5, 0.5]))


def test_valid_x_y_z_energy_object():
    """vector.obj should accept x, y, z, energy"""
    vec = vector.obj(x=1.0, y=2.0, z=3.0, energy=5.0)
    assert vec.x == 1.0
    assert vec.y == 2.0
    assert vec.z == 3.0
    assert vec.energy == 5.0


def test_valid_x_y_z_energy_numpy():
    """vector.array should accept x, y, z, energy"""
    arr = vector.array(
        {
            "x": np.array([1.0, 2.0]),
            "y": np.array([2.0, 3.0]),
            "z": np.array([3.0, 4.0]),
            "energy": np.array([5.0, 6.0]),
        }
    )
    assert np.allclose(arr.x, [1.0, 2.0])
    assert np.allclose(arr.y, [2.0, 3.0])
    assert np.allclose(arr.z, [3.0, 4.0])
    assert np.allclose(arr.energy, [5.0, 6.0])


def test_valid_x_y_z_energy_awkward():
    """vector.Array should accept x, y, z, energy"""
    arr = vector.Array(
        ak.Array(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([2.0, 3.0]),
                "z": np.array([3.0, 4.0]),
                "energy": np.array([5.0, 6.0]),
            }
        )
    )
    assert ak.all(arr.x == ak.Array([1.0, 2.0]))
    assert ak.all(arr.y == ak.Array([2.0, 3.0]))
    assert ak.all(arr.z == ak.Array([3.0, 4.0]))
    assert ak.all(arr.energy == ak.Array([5.0, 6.0]))


def test_valid_x_y_z_energy_zip():
    """vector.zip should accept x, y, z, energy"""
    arr = vector.zip(
        {
            "x": np.array([1.0, 2.0]),
            "y": np.array([2.0, 3.0]),
            "z": np.array([3.0, 4.0]),
            "energy": np.array([5.0, 6.0]),
        }
    )
    assert ak.all(arr.x == ak.Array([1.0, 2.0]))
    assert ak.all(arr.y == ak.Array([2.0, 3.0]))
    assert ak.all(arr.z == ak.Array([3.0, 4.0]))
    assert ak.all(arr.energy == ak.Array([5.0, 6.0]))


def test_valid_px_py_pz_E_object():
    """vector.obj should accept px, py, pz, E"""
    vec = vector.obj(px=1.0, py=2.0, pz=3.0, E=5.0)
    assert vec.px == 1.0
    assert vec.py == 2.0
    assert vec.pz == 3.0
    assert vec.E == 5.0


def test_valid_px_py_pz_E_numpy():
    """vector.array should accept px, py, pz, E"""
    arr = vector.array(
        {
            "px": np.array([1.0, 2.0]),
            "py": np.array([2.0, 3.0]),
            "pz": np.array([3.0, 4.0]),
            "E": np.array([5.0, 6.0]),
        }
    )
    assert np.allclose(arr.px, [1.0, 2.0])
    assert np.allclose(arr.py, [2.0, 3.0])
    assert np.allclose(arr.pz, [3.0, 4.0])
    assert np.allclose(arr.E, [5.0, 6.0])


def test_valid_px_py_pz_E_awkward():
    """vector.Array should accept px, py, pz, E"""
    arr = vector.Array(
        ak.Array(
            {
                "px": np.array([1.0, 2.0]),
                "py": np.array([2.0, 3.0]),
                "pz": np.array([3.0, 4.0]),
                "E": np.array([5.0, 6.0]),
            }
        )
    )
    assert ak.all(arr.px == ak.Array([1.0, 2.0]))
    assert ak.all(arr.py == ak.Array([2.0, 3.0]))
    assert ak.all(arr.pz == ak.Array([3.0, 4.0]))
    assert ak.all(ak.Array([5.0, 6.0]) == arr.E)


def test_valid_px_py_pz_E_zip():
    """vector.zip should accept px, py, pz, E"""
    arr = vector.zip(
        {
            "px": np.array([1.0, 2.0]),
            "py": np.array([2.0, 3.0]),
            "pz": np.array([3.0, 4.0]),
            "E": np.array([5.0, 6.0]),
        }
    )
    assert ak.all(arr.px == ak.Array([1.0, 2.0]))
    assert ak.all(arr.py == ak.Array([2.0, 3.0]))
    assert ak.all(arr.pz == ak.Array([3.0, 4.0]))
    assert ak.all(ak.Array([5.0, 6.0]) == arr.E)


# ============================================================================
# Incomplete azimuthal coordinate pairs
# ============================================================================


def test_incomplete_x_without_y_object():
    """vector.obj should reject x without y"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(x=1.0, z=3.0)


def test_incomplete_x_without_y_numpy():
    """vector.array should reject x without y"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.array(
            {
                "x": np.array([1.0, 2.0]),
                "z": np.array([3.0, 4.0]),
            }
        )


def test_incomplete_x_without_y_awkward():
    """vector.Array should reject x without y"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.Array(
            ak.Array(
                {
                    "x": np.array([1.0, 2.0]),
                    "z": np.array([3.0, 4.0]),
                }
            )
        )


def test_incomplete_x_without_y_zip():
    """vector.zip should reject x without y"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.zip(
            {
                "x": np.array([1.0, 2.0]),
                "z": np.array([3.0, 4.0]),
            }
        )


def test_incomplete_rho_without_phi_object():
    """vector.obj should reject rho without phi"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(rho=1.0, z=3.0)


def test_incomplete_rho_without_phi_numpy():
    """vector.array should reject rho without phi"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.array(
            {
                "rho": np.array([1.0, 2.0]),
                "z": np.array([3.0, 4.0]),
            }
        )


def test_incomplete_rho_without_phi_awkward():
    """vector.Array should reject rho without phi"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.Array(
            ak.Array(
                {
                    "rho": np.array([1.0, 2.0]),
                    "z": np.array([3.0, 4.0]),
                }
            )
        )


def test_incomplete_rho_without_phi_zip():
    """vector.zip should reject rho without phi"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.zip(
            {
                "rho": np.array([1.0, 2.0]),
                "z": np.array([3.0, 4.0]),
            }
        )


# ============================================================================
# Mixed azimuthal coordinate components
# ============================================================================


def test_mixed_x_phi_object():
    """vector.obj should reject x with phi (mixed systems)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(x=1.0, phi=0.5, z=3.0)


def test_mixed_x_phi_numpy():
    """vector.array should reject x with phi (mixed systems)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.array(
            {
                "x": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "z": np.array([3.0, 4.0]),
            }
        )


def test_mixed_x_phi_awkward():
    """vector.Array should reject x with phi (mixed systems)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.Array(
            ak.Array(
                {
                    "x": np.array([1.0, 2.0]),
                    "phi": np.array([0.5, 1.0]),
                    "z": np.array([3.0, 4.0]),
                }
            )
        )


def test_mixed_x_phi_zip():
    """vector.zip should reject x with phi (mixed systems)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.zip(
            {
                "x": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "z": np.array([3.0, 4.0]),
            }
        )


def test_mixed_y_rho_object():
    """vector.obj should reject y with rho (mixed systems)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(y=2.0, rho=1.0, z=3.0)


def test_mixed_y_rho_numpy():
    """vector.array should reject y with rho (mixed systems)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.array(
            {
                "y": np.array([2.0, 3.0]),
                "rho": np.array([1.0, 2.0]),
                "z": np.array([3.0, 4.0]),
            }
        )


def test_mixed_y_rho_awkward():
    """vector.Array should reject y with rho (mixed systems)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.Array(
            ak.Array(
                {
                    "y": np.array([2.0, 3.0]),
                    "rho": np.array([1.0, 2.0]),
                    "z": np.array([3.0, 4.0]),
                }
            )
        )


def test_mixed_y_rho_zip():
    """vector.zip should reject y with rho (mixed systems)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.zip(
            {
                "y": np.array([2.0, 3.0]),
                "rho": np.array([1.0, 2.0]),
                "z": np.array([3.0, 4.0]),
            }
        )


# ============================================================================
# Temporal without proper 3D base
# ============================================================================


def test_temporal_without_longitudinal_object():
    """vector.obj should reject x+y+t (temporal without longitudinal)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(x=1.0, y=2.0, t=5.0)


def test_temporal_without_longitudinal_numpy():
    """vector.array should reject x+y+t (temporal without longitudinal)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.array(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([2.0, 3.0]),
                "t": np.array([5.0, 6.0]),
            }
        )


def test_temporal_without_longitudinal_awkward():
    """vector.Array should reject x+y+t (temporal without longitudinal)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.Array(
            ak.Array(
                {
                    "x": np.array([1.0, 2.0]),
                    "y": np.array([2.0, 3.0]),
                    "t": np.array([5.0, 6.0]),
                }
            )
        )


def test_temporal_without_longitudinal_zip():
    """vector.zip should reject x+y+t (temporal without longitudinal)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.zip(
            {
                "x": np.array([1.0, 2.0]),
                "y": np.array([2.0, 3.0]),
                "t": np.array([5.0, 6.0]),
            }
        )


def test_mass_without_longitudinal_object():
    """vector.obj should reject pt+phi+mass (temporal without longitudinal)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(pt=1.0, phi=0.5, mass=0.5)


def test_mass_without_longitudinal_numpy():
    """vector.array should reject pt+phi+mass (temporal without longitudinal)"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.array(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "mass": np.array([0.5, 0.5]),
            }
        )


def test_mass_without_longitudinal_awkward():
    """vector.Array should reject pt+phi+mass (temporal without longitudinal)"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.Array(
            ak.Array(
                {
                    "pt": np.array([1.0, 2.0]),
                    "phi": np.array([0.5, 1.0]),
                    "mass": np.array([0.5, 0.5]),
                }
            )
        )


def test_mass_without_longitudinal_zip():
    """vector.zip should reject pt+phi+mass (temporal without longitudinal)"""
    with pytest.raises(TypeError, match="duplicate coordinates"):
        vector.zip(
            {
                "pt": np.array([1.0, 2.0]),
                "phi": np.array([0.5, 1.0]),
                "mass": np.array([0.5, 0.5]),
            }
        )


# ============================================================================
# Missing required coordinates
# ============================================================================


def test_only_temporal_object():
    """vector.obj should reject only t (missing spatial coords)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(t=5.0)


def test_only_temporal_numpy():
    """vector.array should reject only t (missing spatial coords)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.array(
            {
                "t": np.array([5.0, 6.0]),
            }
        )


def test_only_temporal_awkward():
    """vector.Array should reject only t (missing spatial coords)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.Array(
            ak.Array(
                {
                    "t": np.array([5.0, 6.0]),
                }
            )
        )


def test_only_temporal_zip():
    """vector.zip should reject only t (missing spatial coords)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.zip(
            {
                "t": np.array([5.0, 6.0]),
            }
        )


def test_only_longitudinal_object():
    """vector.obj should reject only z (missing azimuthal coords)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(z=3.0)


def test_only_longitudinal_numpy():
    """vector.array should reject only z (missing azimuthal coords)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.array(
            {
                "z": np.array([3.0, 4.0]),
            }
        )


def test_only_longitudinal_awkward():
    """vector.Array should reject only z (missing azimuthal coords)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.Array(
            ak.Array(
                {
                    "z": np.array([3.0, 4.0]),
                }
            )
        )


def test_only_longitudinal_zip():
    """vector.zip should reject only z (missing azimuthal coords)"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.zip(
            {
                "z": np.array([3.0, 4.0]),
            }
        )


def test_longitudinal_temporal_without_azimuthal_object():
    """vector.obj should reject z+t without azimuthal coords"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.obj(z=3.0, t=5.0)


def test_longitudinal_temporal_without_azimuthal_numpy():
    """vector.array should reject z+t without azimuthal coords"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.array(
            {
                "z": np.array([3.0, 4.0]),
                "t": np.array([5.0, 6.0]),
            }
        )


def test_longitudinal_temporal_without_azimuthal_awkward():
    """vector.Array should reject z+t without azimuthal coords"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.Array(
            ak.Array(
                {
                    "z": np.array([3.0, 4.0]),
                    "t": np.array([5.0, 6.0]),
                }
            )
        )


def test_longitudinal_temporal_without_azimuthal_zip():
    """vector.zip should reject z+t without azimuthal coords"""
    with pytest.raises(TypeError, match="unrecognized combination"):
        vector.zip(
            {
                "z": np.array([3.0, 4.0]),
                "t": np.array([5.0, 6.0]),
            }
        )
