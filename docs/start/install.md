# Installation

**MedPipe** requires Python 3.12 or later.

## Standard installation

Install the published package from PyPI:

```bash
pip install medpipe
```

Check that the installation worked:

```bash
python -c "import medpipe; print(medpipe.__version__)"
```

## Installing from source (developer setup)

To set up a local development environment and contribute to **MedPipe**, clone the GitHub repository:

```bash
git clone git@github.com:Surgical-Recovery-and-Safety-Lab/medpipe.git
cd medpipe
```

Install the package in editable mode with the development dependencies:

```bash
pip install -e ".[dev]"
```

The `dev` extra includes the test tools (`pytest`) and the `docs` extra (`mkdocs`, linting and type-checking tools). To preview this documentation locally:

```bash
mkdocs serve
```

---

## Next step

Once the package has been installed, try the [quickstart example](quickstart.md).
