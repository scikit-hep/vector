# -*- coding: utf-8 -*-

import numpy as np

import pytest


@pytest.fixture(scope="session")
def ak_HZZ_example():
    skhep_testdata = pytest.importorskip("skhep_testdata")
    uproot = pytest.importorskip("uproot")
    ak = pytest.importorskip("awkward")

    tree = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"]

    x, y, z, t = tree.arrays(["Muon_Px", "Muon_Py", "Muon_Pz", "Muon_E"], how=tuple)

    offsets = x.layout.offsets

    content = ak.layout.RecordArray(
        [
            ak.layout.NumpyArray(ak.values_astype(x.layout.content, "float64")),
            ak.layout.NumpyArray(ak.values_astype(y.layout.content, "float64")),
            ak.layout.NumpyArray(ak.values_astype(z.layout.content, "float64")),
            ak.layout.NumpyArray(ak.values_astype(t.layout.content, "float64")),
        ],
        keys=["x", "y", "z", "t"],
        parameters={"__record__": "LorentzXYZT"},
    )

    # This array is generic: it doesn't know what records labeled "LorentzXYZT" mean.
    return ak.Array(ak.layout.ListOffsetArray64(offsets, content))
