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

## Layout

```
src/
  config.py            constants: box size, snapshot, selection cuts, kNN grid,
                        LH and 1P paths
  data_io.py            find_snapshots()/parse_dir_label() (LH and 1P, via `prefix`),
                         read_bh_catalog()
  selection.py           bolometric luminosity, Eddington ratio; top-fraction,
                          fixed-N luminosity, and fixed-N mass selection
  knn_cdf.py              the kNN-CDF statistic itself
  params.py                 LH/1P parameter table loading + sim_id-keyed alignment
  pipeline.py                run_suite(): every snapshot -> one .npz of summaries
                              (LH or 1P, luminosity- or mass-selected)
  onep.py                     1P label parsing, p<N>-name inference, per-step
                               CDF grouping, trend test
  abundance.py                 remove the "more AGN -> different CDF" trend
                                (top-fraction selection only)
  selection_bias.py             luminosity-cut bias diagnostics + nbh-confound
                                 correction
  sensitivity.py                 scale-resolved + scalar parameter response,
                                  with bootstrap/null, FDR-adjusted significance
  plotting.py                     figures: scale response, sensitivity bar
                                   (with null floor), 1P sweep
notebooks/
  01_agn_luminosity_knn_sensitivity.ipynb   top-10%-by-luminosity run (executed)
  02_fixed_density_agn_knn.ipynb             fixed-number-density run + comparison
  03_mass_selected_control.ipynb             mass- vs. luminosity-selected, same N
  04_1p_parameter_sweep.ipynb                 1P sweep: is feedback swamped by LH
                                               marginalization, or really null?
tests/
  synthetic-data / fake-I/O unit tests for every module above (no simulation
  data required to run these) — including a pipeline.run_suite() integration
  test with find_snapshots/read_bh_catalog monkeypatched
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
`PARAMS_FILE` (and `SIM_PATH_1P`/`PARAMS_FILE_1P` for notebook 04) in
`src/config.py` at your local copy, then run the notebooks in order —
01 → 02 → 03/04 (03 and 04 both depend on 02's saved `.npz` for `N_TARGET`,
not on each other). `pytest` runs independently of any real data:

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
- **1P sweep** (notebook 04): this suite's 1P set turned out to vary 28
  astrophysics parameters (`WindEnergyIn1e51erg`, `RadioFeedbackFactor`, ...),
  not the LH run's 4 lumped ones — a finer decomposition with no 1:1 name
  match, discovered empirically (see `src/onep.py`'s module docstring) after
  an initial wrong assumption about the directory naming convention. Only
  `Omega_m`/`sigma_8` are directly comparable to LH; the 28 astrophysics
  columns are ranked on their own terms (FDR-corrected separately from the
  cosmological pair) for whether *any* of them shows a real trend that a
  literal `A_SN1`-style re-test can't answer. Not yet run against real data
  past the fix — see the notebook's own "Reading this" section.

## Roadmap (not built yet)

- **Why does luminosity selection null out `A_SN1`** when mass-selection at
  the identical N doesn't (notebook 03's biggest open question)? Worth
  checking whether it's specific to `A_SN1` or shows up for other feedback
  parameters at different N/snapshots before reading much into it.
- **Degeneracy**: whether two parameters (e.g. `Omega_m` and `A_AGN1`) leave
  similar-looking imprints the kNN-CDF alone can't tell apart. Lower priority
  until a second parameter shows a real signal (currently only `Omega_m`
  does) — a Fisher/covariance analysis has nothing to act on with one axis.
- **CV set** (27 sims, fiducial parameters, different seeds) as a
  cosmic-variance noise floor, to turn `R(p)` into an interpretable S/N
  rather than a bare number compared to its own permutation null.
- Robustness: does the `Omega_m` result hold across different `N_TARGET`,
  other snapshots/redshifts, and an Eddington-ratio (rather than luminosity)
  selection?
