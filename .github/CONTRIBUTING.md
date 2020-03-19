
## Development install

You should *always* use a virtual environment when developing software. Setup:

```bash
python3 -m venv .env
. .env/bin/activate
pip install -e .[dev,test]
```

If you use conda environments and want to compare against ROOT:

```bash
conda env create
conda activate vector
conda config --env --add channels conda-forge  # Optional
```

You can update the environment with `conda env update`.


## Docs

The documentation is in `/docs`. To rebuild the API docs:

```bash
sphinx-apidoc -o api ../src/vector -M -f
```

## Design

The library is in `src/vector`. The main subpackages are:
* `core`: The numpy ufunct free functions that do most of the calculations
* `common`: The (above) ufuncts wrapped into a mixin class
* `single`: The single-value backend (used by Numba)
* `numpy`: The numpy backend
* `awkward`: The awkward1 backend
* `numba`: The numba backend
* `numba.awkward`: The awkward-numba backend

In the future, you may not need to import so many subpackages. But that's for a later time.

## Adding a method

If you add a new simple method (returns a value), you need to:

1. `core.lorentz.all`: The ufunc goes here.
2. `common.lorentz.all`: The method gets added here. Docstring goes here. This adds `np.errstate` if division is valid.
3.
