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
  config.py            constants: box size, snapshot, selection cuts, kNN grid, paths
  data_io.py            find_snapshots(), read_bh_catalog()
  selection.py           bolometric luminosity, Eddington ratio, top-fraction AGN selection
  knn_cdf.py              the kNN-CDF statistic itself
  params.py                 LH parameter table loading + sim_id-keyed alignment
  pipeline.py                run_suite(): every snapshot -> one .npz of summaries
  abundance.py                 remove the "more AGN -> different CDF" trend
  selection_bias.py             luminosity-cut bias diagnostics + nbh-confound correction
  sensitivity.py                 scale-resolved + scalar parameter response, with bootstrap/null
  plotting.py                     figures for sensitivity_table() output
notebooks/
  01_agn_luminosity_knn_sensitivity.ipynb   the full run, end to end
tests/
  synthetic-data unit tests for knn_cdf, abundance, sensitivity, and the
  sim_id alignment logic (no simulation data required to run these)
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
`PARAMS_FILE` in `src/config.py` (or the notebook's config cell) at your local
copy, then run `notebooks/01_agn_luminosity_knn_sensitivity.ipynb` top to
bottom. `pytest` runs independently of any real data:

```
pip install -r requirements.txt
pytest tests/
```

## Roadmap (not built yet)

This first version answers *whether and where* each of the 6 parameters
imprints on the AGN kNN-CDF, one parameter at a time. The natural next step —
**degeneracy**: whether two parameters (e.g. `Omega_m` and `A_AGN1`) leave
similar-looking imprints that the kNN-CDF alone can't tell apart — is
intentionally out of scope here and will build on top of this pipeline once
the single-parameter results are validated against real data.
