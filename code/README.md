# Supplementary code and data

Reproduces every table and figure of "Solving a second-order energy-stable Cahn–Hilliard scheme by preconditioned descent: global convergence, computable stopping rules, and error estimates".

## Requirements

- Python 3.10 or later
- NumPy, Matplotlib, SciPy
- SymPy (only for the spatial-convergence test)

## Files

| File | Purpose |
| --- | --- |
| `chsolver.py` | The scheme (Cheng–Wang–Wise–Yue 2016, α = 0) and the inner solvers: fixed-K sweeps, relaxed iteration, Armijo line search (c = 1/4), and damped Newton–PCG. It also implements the acceptance tests (T_E), (T_A) and the B-norm tolerance, and counts every real FFT (including CG work). |
| `experiments.py` | Simulation layer, part 1: accuracy, spatial convergence, whole-run solver table, mesh study, effectivity iterates. Writes JSON to `../data/`. |
| `fig_experiments.py` | Simulation layer, part 2: residual histories (Fig. 2a,b), work–precision (Fig. 2c), accepted-iterate test (Fig. 3b), 512² lockstep benchmark (Figs. 4–5), time-step sensitivity. Each output records grid, domain, ε, A, τ, seed, reference solver and norm conventions. |
| `fig_experiments2.py` | Simulation layer, part 3: resolved interface experiments (star and two circles, Fig. 6; unequal droplets, Fig. 7; resolution study, Table 2) and the supplementary seed study (Fig. S2, Table S1). Defines the geometric observables (periodic component count, marching-squares droplet radii, interface displacement). |
| `diagnostics.py` | Diagnostic layer: reads the saved data only and writes `../data/summary.json`, which contains every number quoted in Section 7. |
| `make_figures.py` | Plotting layer: builds `../figs/fig1_accuracy.pdf` … `fig5_structure.pdf` and `../tex/tab_*.tex` from saved data only. |
| `test_chsolver.py` | Unit checks: gradient of J against finite differences, agreement of all inner solvers, the perturbed energy identity, and temporal order. |

## Reproducing the paper

Run from this directory:

```
# Figure 1 (accuracy) and the spatial-convergence numbers
python3 experiments.py time; python3 experiments.py time2; python3 experiments.py armijo time
python3 experiments.py space
# Table 1 (whole-run solver comparison) and Figure 2(d) (mesh study)
python3 experiments.py solver tight; python3 experiments.py solver practical
python3 experiments.py armijo tight; python3 experiments.py armijo practical
python3 experiments.py mesh; python3 experiments.py armijo mesh
# Figure 2(a,b,c), Figure 3
python3 fig_experiments.py resid
python3 fig_experiments.py work
python3 experiments.py sharp; python3 fig_experiments.py accept
# Figures 4 and 5 (512^2 benchmark, ~40 min) and the tau/2 comparison
python3 fig_experiments.py bench
python3 fig_experiments.py tauscale
# resolved interface experiments (Figs. 6-7, Table 2) and supplementary seed study (Fig. S2, Table S1)
python3 fig_experiments2.py star_refine
python3 fig_experiments2.py star_prod 128 0.001
python3 fig_experiments2.py drop_pilot 4
python3 fig_experiments2.py drop_refine 2 0.15,1,2
python3 fig_experiments2.py drop_prod 256 0.001 2 0.15,1,2
python3 fig_experiments2.py seeds; python3 fig_experiments2.py seed_refine
# numbers and plots
python3 diagnostics.py
python3 make_figures.py 1 2 3 4 5 6 7 S2 tables   # S2 -> Figure 8 of the paper (seed study); all tables
```

The two-sided energy test |(R_n(v), v-phi^n)| <= eta D_n(v), equation (5.2) of the paper, is what `chsolver.py` implements (`pert = abs(...)`); Figure 5(d) plots the same ratio.

Plotting conventions: one colour and one marker per inner method (Newton–PCG blue circles, Armijo green squares, relaxed ω=½ orange triangles, linear ω=1 purple diamonds); solid = residual stopping, dashed = fixed sweep count; filled = practical rule, open = tight rule. Phase fields use a diverging map centred at 0 with common limits [−1.2, 1.2]; error magnitudes use a sequential logarithmic map with common limits; out-of-range values are reported in the captions. Curves are vector graphics, fields are rasterized.

Run time is a few hours on two cores; the 512² runs dominate.

Random initial data use NumPy's `default_rng` with fixed seeds. The seed-1 initial field of the 512² benchmark is also archived as `../data/phi0_seed1_512.npy`, so the run can be reproduced even if generator streams change between NumPy versions.


Figure files and paper numbering: fig1_accuracy (Fig. 1), fig2_robustness (Fig. 2), fig3_estimator (Fig. 3), fig4_coarsening (Fig. 4), fig5_structure (Fig. 5), fig6_star (Fig. 6), fig7_droplets (Fig. 7), figS2_seeds (Fig. 8). Tables: tab_accuracy (Table 1), tab_solver (2), tab_resolution (3), tab_seeds (4), tab_summary (5). The former supplementary material is now part of the paper (Section 5 and the appendix); figS1_accuracy_all is superseded by Table 1.
