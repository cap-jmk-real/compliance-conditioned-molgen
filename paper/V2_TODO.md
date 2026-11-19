# V2 TODO (non-blocking findings deferred to a future revision)

These are real but non-blocking items from the paper-review and klaut.pro panel passes.
None are manuscript blockers. The two that would most raise the score need new experiments,
which were out of scope for this preprint (single training run per arm, fixed seed, by brief).

## Needs new experiments (the main score ceiling)
- Train >=3 seeds per arm and report a seed-level paired test with effect size and paired CIs,
  so uncertainty covers training variability and not only sampling at fixed weights.
- Factorial ablations across depth, loss and step budget to attribute the effect.
- Multiple target bands across the spectrum, and a scaling check vs the share of in-band
  training examples.
- Wet-lab or higher-fidelity validation of a sampled subset, to move beyond surrogate scores.

## Writing / reproducibility (no experiments needed, deferred for speed)
- Add a full software-stack line (CUDA, cuDNN, PyTorch, scikit-learn versions) beyond the
  repository requirements files.
- Add an explicit artifact manifest / one-command reproduce script in the repo README.
- Consolidate the repeated "never a target, never a reward" statement to one anchored place.
- Consider citing a canonical SMILES/SELFIES RNN baseline implementation for the GRU LM.
