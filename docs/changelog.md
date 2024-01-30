# Changelog

## Version 1.2

### Version 1.2.0

#### Bug fixes

- fix: result of an infix operation should be demoted to the lowest possible dimension [#413][]
- fix: all infix operations should not depend on the order of arguments [#413][]
- fix: return the correct awkward record when performing an infix operation [#413][]
- fix: respect user defined awkward mixin subclasses and projection classes [#413][]

#### Documentation

- Update `README` and `intro.ipynb` to include the latest developments [#399][]
- docs: add docs for vector.zip [#390][]
- Fix Vector\* mixin's docstring [#404][]

#### Maintenance

- chore: repo review updates [#408][]
- black -> ruff format [#414][]
- chore: migrate to pytest-doctestplus [#416][]

[#390]: https://github.com/scikit-hep/vector/pull/390
[#404]: https://github.com/scikit-hep/vector/pull/404
[#399]: https://github.com/scikit-hep/vector/pull/399
[#408]: https://github.com/scikit-hep/vector/pull/408
[#414]: https://github.com/scikit-hep/vector/pull/414
[#416]: https://github.com/scikit-hep/vector/pull/416
[#413]: https://github.com/scikit-hep/vector/pull/413

## Version 1.1

### Version 1.1.1.post1

#### Maintenance

- chore: support Python 3.12 [#388][]
- Fix CI badge in README and docs [#386][]

[#388]: https://github.com/scikit-hep/vector/pull/388
[#386]: https://github.com/scikit-hep/vector/pull/386

### Version 1.1.1

#### Bug fixes

- fix: `keepdims` in `numpy.sum` should not be `None` [#376][]

#### Maintenance

- chore: remove license string (not standard) [#371][]
- chore: blackend-docs moved [#370][]
- chore: use 2x faster black mirror [#367][]
- chore: clean up VCS versioning [#363][]
- chore: target-version no longer needed by Black or Ruff [#359][]
- chore: ruff moved to astral-sh [#358][]

[#376]: https://github.com/scikit-hep/vector/pull/376
[#371]: https://github.com/scikit-hep/vector/pull/371
[#370]: https://github.com/scikit-hep/vector/pull/370
[#367]: https://github.com/scikit-hep/vector/pull/367
[#363]: https://github.com/scikit-hep/vector/pull/363
[#359]: https://github.com/scikit-hep/vector/pull/359
[#358]: https://github.com/scikit-hep/vector/pull/358

### Version 1.1.0

#### Features

- feat: implement `sum`, `count`, and `count_nonzero` reductions [#347][]

#### Maintenance

- chore: remove Python `3.7` support [#355][]
- chore: use trusted publisher deployment [#354][]
- chore: replace custom definition of np.isclose with numba's np.isclose [#348][]

[#355]: https://github.com/scikit-hep/vector/pull/355
[#354]: https://github.com/scikit-hep/vector/pull/354
[#347]: https://github.com/scikit-hep/vector/pull/347
[#348]: https://github.com/scikit-hep/vector/pull/348

## Version 1.0

### Version 1.0.0

#### Features

- feat: add constructors for `VectorObject3D` and `MomentumObject3D` [#231][]
- feat: add constructors for `VectorObject4D` and `MomentumObject4D` [#232][]
- feat: update `to_Vector3D` to pass new coordinate values [#278][]
- feat: allow passing coordinates to to_Vector-D [#319][]

#### Bug fixes

- fix: better elif conditions for obj \_\_init\_\_ methods [#316][]

#### Documentation

- docs: a readable changelog [#320][]

#### Maintenance

- ci: use numpy~=1.24.0 in pre-commit [#308][]
- fix: update discheck [#305][]
- ci: update number of builds for codecov bot [#314][]
- chore: move to using Ruff [#315][]
- chore: update copyright and license for 2022 and 2023 [#321][]

[#231]: https://github.com/scikit-hep/vector/pull/231
[#232]: https://github.com/scikit-hep/vector/pull/232
[#278]: https://github.com/scikit-hep/vector/pull/278
[#316]: https://github.com/scikit-hep/vector/pull/316
[#308]: https://github.com/scikit-hep/vector/pull/308
[#305]: https://github.com/scikit-hep/vector/pull/305
[#314]: https://github.com/scikit-hep/vector/pull/314
[#315]: https://github.com/scikit-hep/vector/pull/315
[#319]: https://github.com/scikit-hep/vector/pull/319
[#320]: https://github.com/scikit-hep/vector/pull/320
[#321]: https://github.com/scikit-hep/vector/pull/321

## Version 0.11

### Version 0.11.0

#### Features

- Add constructors for `VectorObject2D` and `MomentumObject2D`[#89][]
- Add support for awkward v2 (and keep supporting v1) [#284][]

#### Bug fixes

- `vector.arr` should construct `NumPy` vectors [#254][]
- Development dependency missing [#280][]

#### Documentation

- docs: add a section for talks [#264][]
- docs: fix missing backslash in latex for readme [#285][]
- docs: update changelog.md, PR template, and CONTRIBUTING.md [#275][]
- docs: add a developer guide [#233][]

#### Maintenance

- chore: add PyLint and additional pre-commit hooks [#260][]
- chore: pull request template-Priyadarshi [#271][]
- chore: add issue templates [#267][]
- chore: better and long term fix for flake8-bugbear [#298][]
- chore: bump mypy and revert python-version [#263][]
- chore: fix the failing mypy hook by pinning python-version [#261][]
- chore: ignore flake8 B905 + improve bug report template [#297][]
- chore: minor cleanups [#266][]
- chore: test on `awkward v1.10.0` and add cov to `noxfile` [#256][]
- chore: use Python 3.11! [#282][]
- chore: zenodo badge sync [#269][]
- ci: test notebooks on PRs [#272][]

[#256]: https://github.com/scikit-hep/vector/pull/256
[#254]: https://github.com/scikit-hep/vector/pull/254
[#260]: https://github.com/scikit-hep/vector/pull/260
[#261]: https://github.com/scikit-hep/vector/pull/261
[#263]: https://github.com/scikit-hep/vector/pull/263
[#264]: https://github.com/scikit-hep/vector/pull/264
[#266]: https://github.com/scikit-hep/vector/pull/266
[#267]: https://github.com/scikit-hep/vector/pull/267
[#269]: https://github.com/scikit-hep/vector/pull/269
[#271]: https://github.com/scikit-hep/vector/pull/271
[#233]: https://github.com/scikit-hep/vector/pull/233
[#272]: https://github.com/scikit-hep/vector/pull/272
[#89]: https://github.com/scikit-hep/vector/pull/89
[#275]: https://github.com/scikit-hep/vector/pull/275
[#280]: https://github.com/scikit-hep/vector/pull/280
[#282]: https://github.com/scikit-hep/vector/pull/282
[#285]: https://github.com/scikit-hep/vector/pull/285
[#297]: https://github.com/scikit-hep/vector/pull/297
[#298]: https://github.com/scikit-hep/vector/pull/298
[#284]: https://github.com/scikit-hep/vector/pull/284

## Version 0.10

### Version 0.10.0

#### Maintenance

- Remove Python `3.6` support [#251][]

[#251]: https://github.com/scikit-hep/vector/pull/251

## Version 0.9

### Version 0.9.0

#### Features

- Implements deltaRapidityPhi and deltaRapidityPhi2. [#175][]
- Remove underscores [#192][]
- feat: add git archive support [#244][]

#### Bug fixes

- fix bad values for high (abs) eta [#172][]
- Add custom reprs to awkward coordinate classes [#212][]
- Explicitly set posinf and neginf in nan_to_num so they stay infinite. [#173][]
- Add type checks in constructors [#210][]

#### Documentation

- Add Conda and Zenodo badges to the README-rodrigues [#183][]
- Tests and docs for deltaRapidityPhi [#187][]
- docs: fix intro notebook, and submodule and subpackage index [#191][]
- docs: add codecov badge to README [#203][]
- docs: fix warnings [#193][]
- docs: add docstrings in the `backends.numpy` module [#195][]
- docs: add docstrings in the `backends.object` module [#201][]
- docs: improve the landing page and API docs structure [#204][]
- docs: add docstrings in the `backends.awkward` module [#207][]
- Implement doctests in CI [#211][]
- docs: Add CITATION.cff Citation File Format file [#243][]
- docs: update changelog [#248][]

#### Maintenance

- chore: wheel not required for setuptools PEP 517 (all-repos) [#176][]
- fix: bump black to 22.3.0 due to click 8.1 release [#181][]
- ci: fix a test and update CI to catch errors regularly [#199][]
- chore: fix conda badge and update dependabot [#213][]
- docs: render module level docstrings in documentation [#218][]
- chore: pass repo review [#219][]
- chore: migrate to hatchling [#223][]
- chore: add `codecov.yml` [#229][]
- chore: add `pyproject-fmt` pre-commit hook [#230][]
- chore: remove redundant `tool.check-manifest` from `pyproject.toml` [#235][]
- chore: support `awkward` `v1` and `v2` together [#226][]
- chore: build and test on Python `3.10` and `3.11-dev` [#252][]

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
