# Changelog

## Version 0.10

### Version 0.10.0

- Remove Python `3.6` support [#251][]

[#251]: https://github.com/scikit-hep/vector/pull/251

## Version 0.9

### Version 0.9.0

- Wheel not required for setuptools PEP 517 (all-repos) [#176][]
- Fix bad values for high (abs) eta [#172][]
- Bump black to 22.3.0 due to click 8.1 release [#181][]
- Add Conda and Zenodo badges to the README [#183][]
- Implement deltaRapidityPhi and deltaRapidityPhi2 [#175][]
- Tests and docs for deltaRapidityPhi [#187][]
- Fix intro notebook, and submodule and subpackage index [#191][]
- Fix a test and update CI to catch errors regularly [#199][]
- Remove underscores from `_backend` subpackage and every backend module [#192][]
- Add codecov badge to README [#203][]
- Fix documentation warnings [#193][]
- Add docstrings in the `backends.numpy` module [#195][]
- Add docstrings in the `backends.object` module [#201][]
- Improve the landing page and API docs structure [#204][]
- Add docstrings in the `backends.awkward` module [#207][]
- Implement doctests in CI [#211][]
- Add custom reprs to awkward coordinate classes [#212][]
- Fix conda badge and update dependabot [#213][]
- Render module level docstrings in documentation [#218][]
- Pass repo review https[#219][]
- Explicitly set posinf and neginf in nan_to_num so they stay infinite [#173][]
- Add type checks in constructors https[#210][]
- Migrate to hatchling [#223][]
- Add `codecov.yml` [#229][]
- Add `pyproject-fmt` pre-commit hook [#230][]
- Remove redundant `tool.check-manifest` from `pyproject.toml` [#235][]
- Add git archive support [#244][]
- Add CITATION.cff Citation File Format file [#243][]
- Test `Vector` on `Awkward` `v1` and `v2` together [#226][]
- Build and test on Python `3.10` and `3.11-dev` [#252][]

[#176]: https://github.com/scikit-hep/vector/pull/176
[#172]: https://github.com/scikit-hep/vector/pull/172
[#181]: https://github.com/scikit-hep/vector/pull/181
[#175]: https://github.com/scikit-hep/vector/pull/175
[#187]: https://github.com/scikit-hep/vector/pull/187
[#191]: https://github.com/scikit-hep/vector/pull/191
[#199]: https://github.com/scikit-hep/vector/pull/199
[#192]: https://github.com/scikit-hep/vector/pull/192
[#203]: https://github.com/scikit-hep/vector/pull/203
[#193]: https://github.com/scikit-hep/vector/pull/193
[#195]: https://github.com/scikit-hep/vector/pull/195
[#201]: https://github.com/scikit-hep/vector/pull/201
[#204]: https://github.com/scikit-hep/vector/pull/204
[#207]: https://github.com/scikit-hep/vector/pull/207
[#211]: https://github.com/scikit-hep/vector/pull/211
[#212]: https://github.com/scikit-hep/vector/pull/212
[#213]: https://github.com/scikit-hep/vector/pull/213
[#218]: https://github.com/scikit-hep/vector/pull/218
[#219]: https://github.com/scikit-hep/vector/pull/219
[#173]: https://github.com/scikit-hep/vector/pull/173
[#210]: https://github.com/scikit-hep/vector/pull/210
[#223]: https://github.com/scikit-hep/vector/pull/223
[#229]: https://github.com/scikit-hep/vector/pull/229
[#230]: https://github.com/scikit-hep/vector/pull/230
[#235]: https://github.com/scikit-hep/vector/pull/235
[#244]: https://github.com/scikit-hep/vector/pull/244
[#243]: https://github.com/scikit-hep/vector/pull/243
[#226]: https://github.com/scikit-hep/vector/pull/226
[#252]: https://github.com/scikit-hep/vector/pull/252

## Version 0.8

### Version 0.8.5

- Added boostCM_of to clarify #134, supported by scaleD and negD [#135][]
- Let 'eta' be NaN if 'z' is NaN [#139][]
- Defined dot product without absolute value [#148][]
- Fixed numpy array code examples in documentation [#151][]
- Vector components may be NumpyArrayType or IndexedArrayType in Numba [#162][]
- VectorNumpy pickle support to enable multiprocessing [#163][]
- pre-commit and style cleanup [#164][]

[#135]: https://github.com/scikit-hep/vector/pull/135
[#139]: https://github.com/scikit-hep/vector/pull/139
[#148]: https://github.com/scikit-hep/vector/pull/148
[#151]: https://github.com/scikit-hep/vector/pull/151
[#162]: https://github.com/scikit-hep/vector/pull/162
[#163]: https://github.com/scikit-hep/vector/pull/163
[#164]: https://github.com/scikit-hep/vector/pull/164

### Version 0.8.4

- Allow VectorObject, VectorNumpy, and VectorAwkward to be subclassed by other projects [#128][]

[#128]: https://github.com/scikit-hep/vector/pull/128

### Version 0.8.3

- Fixed Awkward Arrays of momentum vectors in Numba [#112]

[#112]: https://github.com/scikit-hep/vector/pull/112

### Version 0.8.2

- Fixed missing momentum synonyms in CoordinatesAwkward [#84][]
- Added vector.zip [#94][]
- Allowed lowercase e and m for energy and mass [#95][]
- Fixed \_wrap_result for methods called on an ak.Record [#100][]
- Fixed error in calculation of deltaangle [#105][]
- Fixed Awkward version check [#82][]
- Pinned Python version for dis (in tests) [#90][]
- Using myst-parser (in docs) [#91][]

[#82]: https://github.com/scikit-hep/vector/pull/82
[#84]: https://github.com/scikit-hep/vector/pull/84
[#90]: https://github.com/scikit-hep/vector/pull/90
[#91]: https://github.com/scikit-hep/vector/pull/91
[#94]: https://github.com/scikit-hep/vector/pull/94
[#95]: https://github.com/scikit-hep/vector/pull/95
[#100]: https://github.com/scikit-hep/vector/pull/100
[#105]: https://github.com/scikit-hep/vector/pull/105

### Version 0.8.1

- Fix issue importing without Awkward installed [#76][]

[#76]: https://github.com/scikit-hep/vector/pull/76

### Version 0.8.0

First release to PyPI. Initial implementation. Initial features:

- 2D, 3D, and Lorentz vectors
- Single, Array, and Awkward forms
- Supports Numba / Awkward + Numba
- Multiple coordinate systems
- Geometric / momentum versions
- Statically typed

You can currently construct vectors using `obj`/`arr`/`awk` (or
`obj`/`array`/`Array`) for single, NumPy, and Awkward vectors, respectively.
The next version is likely to improve the vector construction process.
