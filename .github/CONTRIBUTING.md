
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
