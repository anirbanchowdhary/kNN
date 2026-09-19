import numpy as np

from src.selection import (
    bolometric_luminosity,
    count_eligible,
    select_brightest_n,
    select_luminous_agn,
    select_most_massive_n,
)


def _catalog(n=200, seed=0):
    rng = np.random.default_rng(seed)
    bh_mass = 10 ** rng.uniform(5, 9, n)          # spans the 1e6 cut
    bh_mdot = 10 ** rng.uniform(-6, -1, n)
    return bh_mass, bh_mdot


def test_select_brightest_n_returns_exactly_n():
    bh_mass, bh_mdot = _catalog()
    for n_target in (5, 20, 50):
        mask = select_brightest_n(bh_mass, bh_mdot, 1e6, n_target)
        assert mask.sum() == n_target


def test_select_brightest_n_picks_the_most_luminous():
    bh_mass, bh_mdot = _catalog()
    mass_cut = 1e6
    n_target = 20

    mask = select_brightest_n(bh_mass, bh_mdot, mass_cut, n_target)
    lbol = bolometric_luminosity(bh_mdot)

    eligible = (bh_mass > mass_cut) & np.isfinite(lbol) & (lbol > 0)
    rejected = eligible & ~mask

    # every selected AGN outshines every eligible-but-rejected one
    assert lbol[mask].min() >= lbol[rejected].max()
    # and every selection respects the mass floor
    assert np.all(bh_mass[mask] > mass_cut)


def test_select_brightest_n_drops_sim_when_too_few_eligible():
    bh_mass = np.array([1e7, 1e8, 1e9])
    bh_mdot = np.array([1e-3, 1e-2, 1e-1])

    assert select_brightest_n(bh_mass, bh_mdot, 1e6, 3).sum() == 3
    # asking for more tracers than exist must drop the sim, not return fewer:
    # a short sample would break the fixed-density guarantee silently
    assert select_brightest_n(bh_mass, bh_mdot, 1e6, 4).sum() == 0


def test_fixed_n_holds_density_constant_across_sims():
    # The whole point of fixed-N: differing underlying populations, identical
    # tracer counts out.
    counts = []
    for seed in range(6):
        bh_mass, bh_mdot = _catalog(n=150 + 40 * seed, seed=seed)
        counts.append(select_brightest_n(bh_mass, bh_mdot, 1e6, 30).sum())

    assert set(counts) == {30}


def test_count_eligible_matches_top_fraction_pool():
    bh_mass, bh_mdot = _catalog()
    mass_cut = 1e6

    n_elig = count_eligible(bh_mass, bh_mdot, mass_cut)
    mask = select_luminous_agn(bh_mass, bh_mdot, mass_cut, top_fraction=0.10)

    # top-fraction keeps roughly that fraction of the eligible pool
    assert 0 < mask.sum() <= n_elig
    assert abs(mask.sum() - 0.10 * n_elig) < 0.05 * n_elig + 2


def test_select_most_massive_n_returns_exactly_n():
    bh_mass, _ = _catalog()
    for n_target in (5, 20, 50):
        mask = select_most_massive_n(bh_mass, 1e6, n_target)
        assert mask.sum() == n_target


def test_select_most_massive_n_picks_the_most_massive():
    bh_mass, _ = _catalog()
    mass_cut = 1e6
    n_target = 20

    mask = select_most_massive_n(bh_mass, mass_cut, n_target)
    eligible = bh_mass > mass_cut
    rejected = eligible & ~mask

    assert bh_mass[mask].min() >= bh_mass[rejected].max()
    assert np.all(bh_mass[mask] > mass_cut)


def test_select_most_massive_n_drops_sim_when_too_few_eligible():
    bh_mass = np.array([1e7, 1e8, 1e9])
    assert select_most_massive_n(bh_mass, 1e6, 3).sum() == 3
    assert select_most_massive_n(bh_mass, 1e6, 4).sum() == 0


def test_select_most_massive_n_holds_density_constant_across_sims():
    counts = []
    for seed in range(6):
        bh_mass, _ = _catalog(n=150 + 40 * seed, seed=seed)
        counts.append(select_most_massive_n(bh_mass, 1e6, 30).sum())
    assert set(counts) == {30}


def test_mass_and_luminosity_selection_can_differ():
    # The whole point of the control: mass-ranking and luminosity-ranking
    # need not pick the same BHs (a low-mass BH can out-accrete a massive,
    # quiescent one), so the two selections are not required to agree.
    rng = np.random.default_rng(7)
    n = 200
    bh_mass = 10 ** rng.uniform(6, 9, n)
    bh_mdot = 10 ** rng.uniform(-6, -1, n)  # independent of mass

    by_mass = select_most_massive_n(bh_mass, 1e6, 30)
    by_lum = select_brightest_n(bh_mass, bh_mdot, 1e6, 30)

    assert by_mass.sum() == by_lum.sum() == 30
    assert not np.array_equal(by_mass, by_lum)
