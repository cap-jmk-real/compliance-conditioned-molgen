"""CI drift test: regenerated LaTeX fragments must match committed paper_3domain/generated/."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "paper_3domain" / "generated"
SCRIPT = ROOT / "paper_3domain/scripts/generate_paper_artifacts.py"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_generated_fragments_match_manifest() -> None:
    manifest_path = GEN / "manifest.json"
    assert manifest_path.is_file(), "Run paper_3domain/scripts/generate_paper_artifacts.py first"
    manifest = json.loads(manifest_path.read_text())

    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    for fname, expected_hash in manifest["files"].items():
        path = GEN / fname
        assert path.is_file(), fname
        got = _sha(path.read_text())
        assert got == expected_hash, f"Drift in {fname}: regenerate and commit, or fix sources"