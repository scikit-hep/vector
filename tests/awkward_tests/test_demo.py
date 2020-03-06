from __future__ import absolute_import

import pytest
from pytest import approx

skhep_testdata = pytest.importorskip("skhep_testdata")
uproot = pytest.importorskip("uproot")

import awkward1 as ak
import numpy as np
import vector.awkward
from vector.awkward import lorentzbehavior
from collections import OrderedDict

data_path = skhep_testdata.data_path


def test_simple_example():
    tree = uproot.open(data_path("uproot-HZZ.root"))["events"]
    x, y, z, t = tree.arrays(
        ["Muon_Px", "Muon_Py", "Muon_Pz", "Muon_E"], outputtype=tuple
    )

    offsets = ak.layout.Index64(x.offsets)
    content = ak.layout.RecordArray(
        [
            ak.layout.NumpyArray(x.content.astype(np.float64)),
            ak.layout.NumpyArray(y.content.astype(np.float64)),
            ak.layout.NumpyArray(z.content.astype(np.float64)),
            ak.layout.NumpyArray(t.content.astype(np.float64)),
        ],
        keys=["x", "y", "z", "t"],
        parameters={"__record__": "LorentzXYZ"},
    )

    # This array is generic: it doesn't know what records labeled "LorentzXYZ" mean.
    example = ak.Array(ak.layout.ListOffsetArray64(offsets, content))
    assert (
        repr(example)
        == '<Array [[{x: -52.9, y: -11.7, ... t: 69.6}]] type=\'2421 * var * struct[["x", "y"...\'>'
    )

    # This new array understands that data labeled "LorentzXYZ" should have the above methods.
    example2 = ak.Array(example, behavior=lorentzbehavior)
    assert (
        repr(example2)
        == "<Array [[Lxyz(-52.9 -11.7 -8.16 54.8), ... ] type='2421 * var * LorentzXYZ'>"
    )

    assert repr(example2[0, 0]) == "Lxyz(-52.9 -11.7 -8.16 54.8)"
    assert type(example2[0, 0]) is vector.awkward.LorentzXYZ

    assert example2[0, 0].mass == approx(0.10559298741436905)
    assert (
        repr(example2.mass)
        == "<Array [[0.106, 0.105], ... [0.104]] type='2421 * var * float64'>"
    )

    # We need a "ak.sizes" function with a simpler interface than this...
    hastwo = ak.count(example2, axis=-1).x >= 2

    assert (
        str(example2[hastwo, 0] + example2[hastwo, 1])
        == "[Lxyz(-15.2 -11 -19.5 94.2), Lxyz(49.8 8.08 48.1 102), ... Lxyz(2.94 18.4 -262 273)]"
    )
    assert (
        str((example2[hastwo, 0] + example2[hastwo, 1]).mass)
        == "[90.2, 74.7, 89.8, 94.9, 92.1, 53.4, 89.8, ... 91.7, 88.8, 101, 91.5, 92.1, 85.4, 76]"
    )
