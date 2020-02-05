def test_import_root():
    import ROOT

    assert ROOT.__file__ is not None
