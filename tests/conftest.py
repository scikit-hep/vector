import pytest
import numpy as np


@pytest.fixture(scope="session")
def ak_HZZ_example():
    skhep_testdata = pytest.importorskip("skhep_testdata")
    uproot = pytest.importorskip("uproot")
    ak = pytest.importorskip("awkward1")

    tree = uproot.open(skhep_testdata.data_path("uproot-HZZ.root"))["events"]

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
    return ak.Array(ak.layout.ListOffsetArray64(offsets, content))
