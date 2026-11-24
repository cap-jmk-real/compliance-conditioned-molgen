# Compiling the JCIM manuscript

```bash
# TeX Live with achemso (tlmgr install achemso)
python3 paper_3domain/scripts/generate_paper_artifacts.py
cd paper_3domain && latexmk -pdf main.tex
```

Bibliography uses the `achemso` document class (ACS/JCIM reference formatting).