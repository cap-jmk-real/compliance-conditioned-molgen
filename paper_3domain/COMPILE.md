# Compiling for MATCH Commun. Math. Comput. Chem.

Author instructions: https://match.pmf.kg.ac.rs/prepare.html

```bash
python3 paper_3domain/scripts/generate_paper_artifacts.py
pytest tests/test_paper_generated_drift.py -q
cd paper_3domain
# TeX Live: latexmk, biber, biblatex
latexmk -pdf -pvc- main.tex   # or: pdflatex + biber + pdflatex ×2
```

- Layout: vendored `match.sty` (follows the official Overleaf MATCH template structure).
- References: `biblatex` numeric style with `sorting=nyt` (alphabetical bibliography, numeric labels). Verify final punctuation against MATCH examples before e-mailing camera-ready PDF to match@pmf.kg.ac.rs.
- No table-of-contents graphic is required by MATCH (JCIM TOC file removed).
- Cover letter draft: `paper_3domain/submission/cover_letter_match.md` (not for publication).