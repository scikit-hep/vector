## Development install

You should _always_ use a virtual environment when developing software. Setup:

```bash
python3 -m venv .env
. .env/bin/activate
pip install -e .[dev,test]
python -m ipykernel install --user --name vector # For notebooks
```

If you use conda environments and want to compare against ROOT:

```bash
conda env create
conda activate vector
conda config --env --add channels conda-forge  # Optional
python -m ipykernel install --user --name vector # For notebooks
```

You can update the environment with `conda env update`.

## Docs

The documentation is in `/docs`. To rebuild the API docs:

```bash
sphinx-apidoc -o api ../src/vector -M -f
```

## Design

The library is in `src/vector`.
