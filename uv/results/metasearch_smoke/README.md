# Meta-search CPU smoke (NOT a result)

A tiny end-to-end run on the build box CPU to show the driver, the gates, the checkpoints
and the report work on the real artefacts: seed-1 plain and steered generators, the fixed
seed-0 surrogate and corpus (249,455 molecules), and the independent holdout predictor.
Budget: 4 guidance, 3 REINVENT and 2 self-training configs, rungs 10 updates / 300 samples
and 30 updates / 500 samples, eta 2, confirm 1,000 samples, `--cpu`, 2.2 min wall-clock.

At 30 optimiser updates and 1,000 samples the in-band differences in `results.md` are
within sampling noise (the standard error of a fraction near 0.25 at n = 1,000 is about
0.014) and one trial per config: do not read a ranking into it. The Spearman line there
has n = 5. The purpose is the plumbing.

`cost_probe_box_cpu.json` is the output of `python -m uv.metasearch_cost` for the default
engine pilot schedule, measured on the same shared CPU while other jobs were running.
