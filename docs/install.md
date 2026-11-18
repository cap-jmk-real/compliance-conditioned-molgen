# Install

Python 3.13. The root `requirements.txt` is pinned to versions that publish 3.13 wheels. The `uv/` experiment needs PyTorch, and the CPU build is enough for the tests and the smoke run.

<!-- ci:run -->
```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
grep -vE '^torch([[:space:]]|$)' uv/requirements.txt > uv-requirements-no-torch.txt
pip install -r uv-requirements-no-torch.txt
```

The `grep` keeps pip from replacing the CPU wheel with the larger CUDA build named in `uv/requirements.txt`. On a GPU machine, `pip install -r uv/requirements.txt` is fine.

Re-fetching the ECHA pages additionally needs Playwright with a headed Chrome at `/usr/bin/google-chrome` and a display. See `DATA.md`.

## Build these docs

<!-- ci:run -->
```bash
pip install -r requirements-docs.txt
mkdocs build --strict
```

The site is not deployed. CI checks that it builds.
