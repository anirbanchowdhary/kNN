# AGN kNN-CDF sensitivity to cosmology and feedback

Rebuilt from scratch (a prior version of this project exists in git history
on `main`, kept only as a reference for data layout and known pitfalls).

## Question

Using **luminosity-selected AGN** (top 10% by bolometric luminosity, above a
`1e6 Msun` base black-hole-mass floor) from a single snapshot
(`snap=50`) of the **CAMELS-IllustrisTNG L25n256 Latin-Hypercube (LH)** suite,
how does the **kNN-CDF** clustering statistic (k = 1, 2, 4) respond to each of
the 6 LH parameters:

- **Cosmological**: `Omega_m`, `sigma_8`
- **Astrophysical (feedback)**: `A_SN1`, `A_AGN1`, `A_SN2`, `A_AGN2`

and *at what spatial scale* r does that response show up?

## Data assumptions

Verified against the [CAMELS data organization docs](https://camels.readthedocs.io/en/latest/organization.html):
CAMELS data on disk (and on Globus / the public URL / the Rusty Cluster
mount) is organized `<Type>/<Suite>/<Generation>/<Set>/<Realization>`:

```
Sims/IllustrisTNG/L25n256/LH/LH_<0..999>/snapshot_<###>.hdf5
Parameters/IllustrisTNG/L25n256/LH/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt
```

- `SUITE="IllustrisTNG"`, `GENERATION="L25n256"` (25 Mpc/h box, matching
  `BOXSIZE`), `SET_NAME="LH"` (1000 simulations, `LH_0`..`LH_999`) — all in
  `src/config.py`, combined into `SIM_PATH`/`PARAMS_FILE` there. Set
  `DATA_ROOT` to wherever your local `Sims/`/`Parameters/` folders live, or
  override `SIM_PATH`/`PARAMS_FILE` directly (or pass
  `run_suite(sim_path=..., ...)` / `load_params(params_file=...)`) if your
  layout differs.
- Snapshot files are `snapshot_<snap:03d>.hdf5` with black holes in the
  `PartType5` HDF5 group (`Coordinates`, `BH_Mass`, `BH_Mdot`); `Header`
  carries `HubbleParam`. This is the **post-2024 naming** (CAMELS renamed
  `snap_###.hdf5` -> `snapshot_###.hdf5` and standardized snapshot numbering
  to 91 steps, 000=z=15 .. 090=z=0, across all suites). If your local data
  predates that reorganization it will use the old `snap_###.hdf5` naming
  and a shorter 34-snapshot range instead — `src/data_io.py:find_snapshots`
  would need updating for that case.
- The parameter table (`CosmoAstroSeed_<suite>_<generation>_<set>.txt`) is
  whitespace-delimited; its first column is `LH_<id>` and the other columns
  are the 6 parameters above plus the random seed.
- **Galaxies** (notebook 06) come from SubFind group/subhalo catalogs:
  `groups_<snap:03d>.hdf5` inside the same per-realization directory as
  the particle snapshot (`GROUPS_PATH = SIM_PATH` in `src/config.py`) —
  **confirmed** against real data (notebook 06's discovery cell); CAMELS's
  general docs describe "Groups" as a logically separate data type, but
  for this account it isn't a separate path. The `Subhalo` HDF5 group
  holds `SubhaloPos` and `SubhaloMassType` (6 columns, index 4 = stellar
  mass), also confirmed; `SubhaloFlag` is **absent** in this dataset, so
  `read_galaxy_catalog` falls back to treating every subhalo as
  non-spurious. `SubhaloPos` can land marginally outside `[0, BOXSIZE)`
  for a subhalo straddling the periodic boundary (2/1000 real LH sims hit
  this on the first run, rejected outright by `scipy`'s boxsize-aware
  `cKDTree`) — `data_io.wrap_periodic` corrects for it.

## Layout

```
src/
  config.py            constants: box size, snapshot, selection cuts, kNN grid,
                        LH, 1P, CV, and Groups (galaxy) paths
  data_io.py            find_snapshots()/parse_dir_label()/read_bh_catalog()
                         (LH, 1P, CV, via `prefix`); find_group_catalogs()/
                         read_galaxy_catalog() for the galaxy tracer
  selection.py           bolometric luminosity, Eddington ratio; top-fraction,
                          fixed-N luminosity, and fixed-N mass selection (BH);
                          fixed-N stellar-mass selection (galaxy)
  knn_cdf.py              the kNN-CDF statistic itself
  params.py                 LH/1P/CV parameter table loading + sim_id-keyed alignment
  pipeline.py                run_suite(): every snapshot -> one .npz of summaries
                              (LH, 1P, or CV; luminosity- or mass-selected AGN);
                              run_galaxy_suite(): the same, over group catalogs
  onep.py                     1P label parsing, p<N>-name inference, per-step
                               CDF grouping, trend test
  cv.py                        CV-set cosmic-variance noise floor (per-bin std,
                                scalar RMS, signal-to-noise)
  abundance.py                 remove the "more AGN -> different CDF" trend
                                (top-fraction selection only)
  selection_bias.py             luminosity-cut bias diagnostics + nbh-confound
                                 correction
  sensitivity.py                 scale-resolved + scalar parameter response,
                                  with bootstrap/null, FDR-adjusted significance
  complementarity.py              cross-tracer per-bin correlation (AGN vs.
                                   galaxies): redundant vs. complementary information
  fisher.py                       Fisher-matrix precision forecast: LH-regression
                                   response + CV-noise covariance -> marginalized
                                   (Omega_m, sigma_8) constraint, per tracer and combined
  plotting.py                     figures: scale response, sensitivity bar
                                   (with null floor), 1P sweep, cross-tracer correlation,
                                   Fisher confidence ellipses
notebooks/
  01_agn_luminosity_knn_sensitivity.ipynb   top-10%-by-luminosity run (executed)
  02_fixed_density_agn_knn.ipynb             fixed-number-density run + comparison
  03_mass_selected_control.ipynb             mass- vs. luminosity-selected, same N
  04_1p_parameter_sweep.ipynb                 1P sweep: is feedback swamped by LH
                                               marginalization, or really null?
  05_cv_noise_floor.ipynb                     CV set: cosmic-variance noise floor,
                                               checked against both the LH Omega_m
                                               signal and the 1P sweep's ranking
  06_galaxy_agn_complementarity.ipynb          galaxies vs. AGN: same-parameter
                                                sensitivity comparison + direct
                                                cross-tracer correlation check
  07_fisher_forecast.ipynb                      Fisher forecast: how tightly could
                                                 (Omega_m, sigma_8) be constrained,
                                                 per tracer and (naively) combined
tests/
  synthetic-data / fake-I/O unit tests for every module above (no simulation
  data required to run these) — including pipeline.run_suite()/run_galaxy_suite()
  integration tests with find_snapshots/read_bh_catalog/find_group_catalogs/
  read_galaxy_catalog monkeypatched
```

## Design choices worth knowing about

- **Scale-resolved, not just scalar.** The parameter response is kept per
  radial bin (shape `(n_k, n_r)`) rather than immediately collapsed to one
  RMS number — the radius axis is the one part of this statistic with direct
  physical meaning, and collapsing it early throws away *where* a parameter
  imprints. The scalar `R(p)` in `sensitivity_table()`'s output is derived
  from the same array (`sqrt(mean(diff**2))`), not computed separately —
  `tests/test_sensitivity.py` checks this consistency directly.
- **Bootstrap CI + permutation null, not point estimates.** Every response
  (scale-resolved and scalar) carries a bootstrap confidence interval
  (resample simulations) and is checked against a permutation-null floor
  (shuffle parameter labels), so "parameter p matters at scale r" is a
  statistical claim, not a read off a noisy line.
- **sim_id-keyed alignment everywhere.** `src/params.align_to_params` is the
  single place a `theta` table gets matched to a `sim_ids` array, and it
  asserts the result is in that exact order. Nothing else in the pipeline is
  allowed to combine per-sim arrays positionally. This exists because a prior
  version of this project had a real bug here: two `.npz` files with
  differently-ordered `sim_ids` got zipped together via `np.isin`-then-mask,
  which preserved each array's *own* order rather than a shared one, and
  silently mixed up which residual vector belonged to which simulation.
- **Fixed number density is the preferred selection.** `run_suite(n_target=N)`
  takes exactly the N brightest AGN per simulation, so tracer density is
  constant across the suite by construction. This matters because the
  kNN-CDF depends on density through a Poisson form that is strongly
  *nonlinear* in n, while `remove_abundance` only removes a trend linear in
  `log10(n)` — the leftover leaks into any parameter correlated with n. In
  the first real run `Omega_m` correlated with AGN count at Spearman
  ρ = 0.98, which makes "does `Omega_m` change clustering or just
  abundance?" unanswerable under the top-fraction selection. Fixed-N removes
  the confound at the source; the cost is that sparse simulations get
  dropped, which makes whole-sim retention the bias to watch instead
  (`diagnose_retention_bias`).
- **Significance comes from the permutation null, never the error bars.**
  `R(p)` is an RMS and therefore positive-definite, so its bootstrap CI
  excludes zero even for a parameter with no effect at all. Read `q_value`
  (Benjamini–Hochberg FDR across the 6 parameters — with 6 tests there's a
  ~26% chance of a spurious p < 0.05) or compare `R_obs` to `null_floor`.
  `plot_sensitivity_bar` draws that floor and fades non-significant bars.
- **Selection-bias diagnostics are not optional for this selection.**
  Because the tracer is *luminosity*-selected (not mass-only), the number of
  AGN surviving the cut in a given simulation can itself correlate with a
  parameter (e.g. more structure formation -> more luminous AGN). If that
  happens, `remove_abundance()` (which regresses each CDF bin against
  `log10(n_agn)`) can remove part of a parameter's real signal along with the
  abundance trend it's meant to strip. `selection_bias.py` diagnoses this
  (`diagnose_retention_bias` for whole-sim dropout,
  `diagnose_nbh_confound` for the within-sim count trend) and provides a
  correction (`residualize_theta_on_nbh`) to apply only where flagged — the
  notebook runs both diagnostics and applies the correction automatically
  where needed.

## Running

No CAMELS simulation data ships with this repo. Point `SIM_PATH` /
`PARAMS_FILE` (`SIM_PATH_1P`/`PARAMS_FILE_1P` for notebook 04,
`SIM_PATH_CV`/`PARAMS_FILE_CV` for notebook 05, `GROUPS_PATH` for notebook
06) in `src/config.py` at your local copy, then run the notebooks in order
— 01 → 02 → 03/04/05/06 (03, 04, 05, and 06 all depend on 02's saved
`.npz` for `N_TARGET`, not on each other). `pytest` runs independently of
any real data:

```
pip install -r requirements.txt
pytest tests/
```

## Findings so far (real-data runs, snap=50)

- **Top-10%-by-luminosity** (notebook 01): `Omega_m` and `sigma_8` both looked
  significant.
- **Fixed number density** (notebook 02, the corrected selection): `Omega_m`'s
  signal more than *doubled* (R: 0.0079 → 0.0207) — the top-fraction result
  was, if anything, an underestimate, because its abundance correction
  couldn't fully separate `Omega_m` from its ρ=0.98 correlation with AGN
  count. Every other parameter's apparent signal *shrank* and none remain
  significant after FDR correction — `sigma_8`'s notebook-01 significance
  doesn't survive. **Only `Omega_m` is a real signal in the LH set.**
- **Mass-selected control** (notebook 03, real-data run): mass-selected
  `Omega_m` (R=0.0192) closely matches AGN's (R=0.0207, ratio 0.93) —
  consistent with the AGN kNN-CDF substantially tracing halo mass, which
  luminosity selection doesn't add much to. But **`A_SN1` is significant
  under mass-selection (R=0.0047, q=0.0015) and null under
  luminosity-selection (R=0.00097, q=0.85)**, a 4.8× ratio — luminosity
  selection may be actively washing out an SN-feedback signal that's visible
  in raw mass-selected clustering. Worth a closer look, not yet explained.
- **1P sweep** (notebook 04, real-data run): this suite's 1P set turned out
  to vary 28 astrophysics parameters (`WindEnergyIn1e51erg`,
  `RadioFeedbackFactor`, ...), not the LH run's 4 lumped ones — a finer
  decomposition with no 1:1 name match, discovered empirically (see
  `src/onep.py`'s module docstring) after an initial wrong assumption about
  the directory naming convention, plus a real label-grammar bug
  (`p10_1`-style labels) fixed along the way. Only `Omega_m`/`sigma_8` are
  directly comparable to LH. Also fixed here: `scipy.stats.spearmanr`'s
  default p-value is badly wrong at this sample size (n≈5 per parameter) —
  an *exact* permutation p-value replaced it. Honest result: **nothing
  clears FDR-corrected significance at n≈5**, not even `Omega_m` — a
  small-n limitation of the test, not evidence the LH `Omega_m` signal is
  wrong (see notebook 05).
- **CV noise floor** (notebook 05, real-data run): the pure seed-to-seed
  scatter (27 fiducial realizations) gives a direct physical noise floor,
  independent of any permutation test. `Omega_m`'s LH response sits at
  ~1.3x that floor — modest but real, and comfortably above 1 even though
  its 1P permutation q-value doesn't clear significance, which is exactly
  what "the small-n permutation test is the limiting factor, not a weak
  `Omega_m` signal" looks like. Ranking all 28 1P astrophysics parameters
  by CV signal-to-noise, `n_s` (spectral index) comes out highest (1.44x,
  mildly surprising), `Omega0` second (0.91x) — nothing else decisively
  clears the noise.
- **Galaxy tracer** (notebook 06, real-data run, 1000/1000 LH simulations
  have group catalogs): fixed-N galaxies (`N_TARGET_GAL=262`, 5th
  percentile, **950/1000 sims retained** after the `wrap_periodic` fix
  below — matches AGN's own retention exactly) show `Omega_m` significant
  (R=0.0233, q≈0) as expected, but also **`sigma_8` significant
  (R=0.0047, q=0.0075)** — a parameter that is *not* significant for AGN
  at the matched fixed-N selection (R=0.0019, q=0.62). This is the first
  real evidence of a second significant parameter anywhere in this
  project, and it shows up specifically in the tracer comparison, not the
  LH set alone. The direct cross-tracer check (`bin_correlation`, section
  6) finds galaxy and AGN clustering fluctuations are correlated but not
  redundant — median |correlation| = 0.69, max = 0.92 across the 929
  simulations common to both runs — consistent with both substantially
  tracing the same large-scale structure while still each carrying
  information the other doesn't (galaxies' `sigma_8` sensitivity being
  the clearest example). Two real simulations (`LH_15`, `LH_263`)
  initially failed with a scipy periodic-box error from `SubhaloPos`
  landing marginally outside `[0, BOXSIZE)`; fixed by wrapping positions
  in `read_galaxy_catalog` (`data_io.wrap_periodic`) and **confirmed
  recovered** on re-run (948→950/1000), with the results above essentially
  unchanged, as expected for a 2/1000-simulation fix.
- **Fisher forecast** (notebook 07, real-data run): turns the LH response
  + CV noise floor into a precision forecast for `(Omega_m, sigma_8)` —
  not a predictor of either parameter's value for a specific simulation
  (that needs a regression/emulator, still on the roadmap below), but an
  answer to "how tightly could this statistic constrain them, in
  principle". The response is estimated by a proper multivariate
  regression against all 6 LH parameters at once (`fisher.linear_response`,
  controlling for the other 5 — not `sensitivity_table`'s marginal
  quartile contrast), and the noise comes from each tracer's own CV run —
  which meant building galaxy CV support for the first time
  (`GROUPS_PATH_CV`), since notebook 05 only ever ran AGN over CV; that
  first real galaxy CV run (27/27 realizations retained at N=262) gives a
  galaxy RMS noise floor of 0.0187, close to but a little above AGN's
  0.0157. **Real numbers**: `sigma(Omega_m)` = 0.046 (AGN) / 0.053
  (galaxy) / 0.025 (naive combined); `sigma(sigma_8)` = 0.278 (AGN) /
  0.226 (galaxy) / 0.119 (naive combined) — galaxies forecast a
  meaningfully tighter `sigma_8` constraint than AGN alone, matching
  notebook 06's significance finding, and combining tightens both axes
  ~1.8-1.9x over the best single tracer (1-sigma ellipse area shrinks to
  27% of the best single tracer's). Two real simplifications, both
  documented in `src/fisher.py`'s module docstring and the notebook's
  "Reading this": only the diagonal of the noise covariance is used (CV
  has too few realizations to invert a ~150-bin empirical covariance —
  33/150 AGN bins and 23/150 galaxy bins were pinned-CDF zero-variance
  and dropped before the real forecast above), and the "combined
  AGN+galaxy" forecast naively sums Fisher matrices, which assumes
  independence that notebook 06 already showed is false (median
  cross-tracer correlation 0.69) — so the combined numbers above are a
  best-case upper bound on the value of combining, not a rigorous joint
  constraint. Also fixed along the way: `run_galaxy_suite` used a fixed
  `"galaxy_knn_..."` output filename regardless of `dir_prefix`, so this
  notebook's CV run at the same N as the LH run would have silently
  overwritten it — now tagged per-prefix like `run_suite` already was,
  with a regression test.

## Roadmap (not built yet)

- **Why does luminosity selection null out `A_SN1`** when mass-selection at
  the identical N doesn't (notebook 03's biggest open question)? Worth
  checking whether it's specific to `A_SN1` or shows up for other feedback
  parameters at different N/snapshots before reading much into it.
- **A rigorous combined (AGN+galaxy) forecast**: needs the AGN-galaxy
  cross-covariance (running both tracers over the *same* CV realizations
  and measuring their joint scatter), not just each tracer's own CV run —
  the naive independent-sum in notebook 07 is only an upper bound.
- **Point-prediction of `Omega_m`/`sigma_8`** from a measured kNN-CDF: a
  regression/emulator (e.g. random forest or Gaussian process) trained on
  the LH suite's `(summary, theta)` pairs, validated on a held-out split —
  a different tool from the Fisher forecast above, which only says how
  precise such a predictor *could* be, not what it would actually predict.
- Robustness: does the `Omega_m` result hold across different `N_TARGET`,
  other snapshots/redshifts, and an Eddington-ratio (rather than luminosity)
  selection?
