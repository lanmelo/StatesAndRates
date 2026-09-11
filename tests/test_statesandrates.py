"""Tests of the states and rates analyses."""

import dataclasses

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt

from statesandrates import (
    Graph,
    Mutation,
    Scheme,
    Trajectory,
    absorption_probabilities,
    barrier_energies,
    coarse_grain,
    cycle_affinity,
    dwell_diagram,
    dwell_rates,
    energy_diagram,
    equilibrium,
    fluxes,
    free_energies,
    integrate,
    is_detailed_balanced,
    is_lumpable,
    mean_first_passage_times,
    mean_residence_times,
    passage_rates,
    propagate,
    relaxation_rates,
    relaxation_times,
    simulate,
    state_diagram,
    thermal_energy,
)

matplotlib.use("Agg")

RT = thermal_energy()


def two_state() -> Graph:
    """Two states, three times more likely to be in the second."""
    return Graph({("a", "b"): 3.0, ("b", "a"): 1.0})


def landscape(
    energies: dict[str, float], barriers: dict[tuple[str, str], float]
) -> Graph:
    """Graph built from a free energy landscape in kcal/mol."""
    return Graph(
        {
            (source, target): np.exp(-(barrier - energies[source]) / RT)
            for pair, barrier in barriers.items()
            for source, target in (pair, pair[::-1])
        },
        states=tuple(energies),
    )


def three_state_cycle(driven: bool = False) -> Graph:
    """Cycle of three states, optionally with a net circulation."""
    rates = {
        ("a", "b"): 2.0,
        ("b", "a"): 1.0,
        ("b", "c"): 3.0,
        ("c", "b"): 4.0,
        ("c", "a"): 5.0,
        ("a", "c"): 1.5 if driven else 2.0 * 3.0 * 5.0 / (1.0 * 4.0),
    }
    return Graph(rates)


def test_rejects_inconsistent_graphs() -> None:
    with pytest.raises(ValueError, match="positive"):
        Graph({("a", "b"): 0.0})
    with pytest.raises(ValueError, match="themselves"):
        Graph({("a", "a"): 1.0, ("a", "b"): 1.0})
    with pytest.raises(ValueError, match="exactly one"):
        Graph({("a", "b"): 1.0}, log_rates={("a", "b"): 1.0})
    with pytest.raises(KeyError, match="unknown state"):
        two_state().index("c")


def test_equilibrium_of_two_states() -> None:
    assert np.allclose(equilibrium(two_state()), [0.25, 0.75])


@pytest.mark.parametrize("driven", [False, True])
def test_equilibrium_is_stationary(driven: bool) -> None:
    graph = three_state_cycle(driven)
    stationary = equilibrium(graph)
    assert np.isclose(stationary.sum(), 1)
    assert np.allclose(graph.generator().T @ stationary, 0, atol=1e-12)


def test_detailed_balance_detects_a_driven_cycle() -> None:
    assert is_detailed_balanced(three_state_cycle())
    assert not is_detailed_balanced(three_state_cycle(driven=True))
    reversible = Graph({("a", "b"): 1.0, ("b", "a"): 1.0})
    assert is_detailed_balanced(reversible)
    assert not is_detailed_balanced(Graph({("a", "b"): 1.0}))
    with pytest.raises(ValueError, match="detailed balance"):
        barrier_energies(three_state_cycle(driven=True))


def test_free_energies_recover_the_landscape() -> None:
    energies = {"a": 0.0, "b": -2.0, "c": 1.5}
    barriers = {("a", "b"): 4.0, ("b", "c"): 6.0}
    graph = landscape(energies, barriers)
    assert np.allclose(
        free_energies(graph, reference="a"), list(energies.values())
    )
    heights = barrier_energies(graph, reference="a")
    assert np.allclose(heights, heights.T, equal_nan=True)
    assert np.isclose(heights[0, 1], barriers[("a", "b")])
    assert np.isclose(heights[1, 2], barriers[("b", "c")])
    assert np.isnan(heights[0, 2])


def test_barrier_prefactor_shifts_every_barrier() -> None:
    graph = three_state_cycle()
    shifted = barrier_energies(graph, prefactor=10.0) - barrier_energies(graph)
    assert np.allclose(shifted[np.isfinite(shifted)], RT * np.log(10.0))


@pytest.mark.parametrize("driven", [False, True])
def test_propagate_relaxes_to_equilibrium(driven: bool) -> None:
    graph = three_state_cycle(driven)
    amounts = propagate(graph, {"a": 7.0}, [0.0, 0.1, 1e3])
    assert np.allclose(amounts[0], [7.0, 0.0, 0.0])
    assert np.allclose(amounts.sum(axis=1), 7.0)
    assert np.allclose(amounts[-1], 7.0 * equilibrium(graph))


def test_propagate_matches_a_two_state_solution() -> None:
    graph = two_state()
    times = np.array([0.0, 0.05, 0.3, 2.0])
    amounts = propagate(graph, {"a": 1.0}, times)
    decay = np.exp(-4.0 * times)  # k_ab + k_ba
    assert np.allclose(amounts[:, 0], 0.25 + 0.75 * decay)


def test_relaxation_rates_of_two_states() -> None:
    graph = two_state()
    assert np.allclose(relaxation_rates(graph), [0.0, 4.0])
    assert np.allclose(relaxation_times(graph), [0.25])
    with pytest.raises(ValueError, match="detailed balance"):
        relaxation_rates(three_state_cycle(driven=True))


def test_propagate_is_stable_over_a_wide_energy_range() -> None:
    graph = landscape(
        {"a": 0.0, "b": -12.0, "c": 8.0}, {("a", "b"): 5.0, ("b", "c"): 14.0}
    )
    stationary = equilibrium(graph)
    assert stationary.min() < 1e-8
    amounts = propagate(graph, stationary, [0.0, 1.0, 1e6])
    assert np.allclose(amounts, stationary, rtol=1e-6)


def test_propagate_is_exact_for_stiff_rates() -> None:
    # A microscopic step 1e5 times faster than the observable kinetics,
    # which overruns the squaring budget of a matrix exponential
    graph = Graph(
        {
            ("free", "testing"): 4.4e-3,
            ("testing", "free"): 1e2,
            ("testing", "bound"): 5.7e2,
            ("bound", "testing"): 6e-3,
        }
    )
    times = np.array([0.0, 2e2, 1e4, 1e9])
    amounts = propagate(graph, {"free": 1.0}, times)
    assert np.isfinite(amounts).all()
    assert np.allclose(amounts.sum(axis=1), 1.0)
    assert np.allclose(amounts[-1], equilibrium(graph))
    # The observable kinetics are a single exponential at k_a + k_d
    k_a = 1 / mean_first_passage_times(graph, "bound")[graph.index("free")]
    k_d = 1 / mean_first_passage_times(graph, "free")[graph.index("bound")]
    decay = np.exp(-(k_a + k_d) * times)
    free = (k_d + k_a * decay) / (k_a + k_d)
    assert np.allclose(amounts[:, 0], free, atol=1e-4)


def test_propagate_takes_reversibility_on_trust() -> None:
    graph = three_state_cycle()
    times = np.array([0.0, 0.3, 1e3])
    expected = propagate(graph, {"a": 1.0}, times)
    traced = jax.jit(
        lambda g: propagate(g, {"a": 1.0}, times, reversible=True)
    )(graph)
    assert np.allclose(traced, expected)
    driven = three_state_cycle(driven=True)
    assert np.allclose(
        propagate(driven, {"a": 1.0}, times, reversible=False),
        propagate(driven, {"a": 1.0}, times),
    )


def test_integrate_matches_propagate_for_constant_rates() -> None:
    graph = three_state_cycle()
    times = np.linspace(0.0, 5.0, 11)
    exact = propagate(graph, {"a": 1.0}, times)
    solved = integrate(lambda t: graph, {"a": 1.0}, times)
    assert np.allclose(exact, solved, atol=1e-6)


def test_integrate_follows_a_time_dependent_rate() -> None:
    graph = two_state()

    # Association switched on at t = 1, when relaxation should start
    def switched(time: jax.Array) -> Graph:
        log_rates = graph.log_rates.at[0, 1].add(jnp.log(time > 1.0))
        return graph.with_log_rates(log_rates)

    amounts = integrate(switched, {"a": 1.0}, [0.0, 1.0, 100.0])
    assert np.allclose(amounts[1], [1.0, 0.0], atol=1e-6)
    assert np.allclose(amounts[2], equilibrium(graph), atol=1e-6)


def test_mutations_perturb_rates_consistently() -> None:
    graph = three_state_cycle()
    destabilized = graph.mutate(Mutation(states={"b": 1.5}))
    assert is_detailed_balanced(destabilized)
    energies = free_energies(destabilized, reference="a") - free_energies(
        graph, reference="a"
    )
    assert np.allclose(energies, [0.0, 1.5, 0.0])

    slowed = graph.mutate(Mutation(barriers={("a", "b"): 2.0}))
    assert is_detailed_balanced(slowed)
    assert np.allclose(equilibrium(slowed), equilibrium(graph))
    factor = np.exp(-2.0 / RT)
    assert np.isclose(slowed.rates[0, 1], graph.rates[0, 1] * factor)
    assert np.isclose(slowed.rates[1, 0], graph.rates[1, 0] * factor)


def test_mutations_add() -> None:
    graph = three_state_cycle()
    first = Mutation(states={"b": 1.0}, barriers={("a", "b"): 0.5}, name="x")
    second = Mutation(states={"b": -0.5}, factors={("b", "c"): 3.0}, name="y")
    combined = first + second
    assert combined.name == "x + y"
    assert np.allclose(
        graph.mutate(combined).log_rates, graph.mutate(first, second).log_rates
    )
    assert np.allclose(
        graph.mutate(Mutation(barriers={("a", "b"): 1.0})).log_rates,
        graph.mutate(Mutation(barriers={("b", "a"): 1.0})).log_rates,
    )


def test_subclasses_survive_copies() -> None:
    class Labelled(Graph):
        """A graph that carries the sequence its rates belong to."""

        sequence: str

    @dataclasses.dataclass(frozen=True)
    class Scored(Mutation):
        score: float = 0.0

    graph = Labelled({("a", "b"): 2.0, ("b", "a"): 1.0})
    graph.sequence = "CACGTG"
    mutant = graph.mutate(Mutation(states={"a": 1.0}))
    assert isinstance(mutant, Labelled)
    assert mutant.sequence == "CACGTG"
    assert not np.allclose(mutant.log_rates, graph.log_rates)

    combined = Scored(states={"a": 1.0}, score=3.0) + Mutation(
        states={"a": 0.5}
    )
    assert isinstance(combined, Scored)
    assert combined.score == 3.0
    assert combined.states == {"a": 1.5}


def test_mean_first_passage_times_of_two_states() -> None:
    graph = two_state()
    assert np.allclose(mean_first_passage_times(graph, "b"), [1 / 3, 0.0])
    assert np.allclose(mean_first_passage_times(graph, ["a", "b"]), [0.0, 0.0])


def test_coarse_graining_preserves_equilibrium() -> None:
    graph = landscape(
        {"free": 0.0, "search": -1.0, "bound": -3.0},
        {("free", "search"): 3.0, ("search", "bound"): 2.0},
    )
    groups = {"unbound": ["free"], "occupied": ["search", "bound"]}
    macro = coarse_grain(graph, groups)
    assert macro.states == ("unbound", "occupied")
    assert is_detailed_balanced(macro)
    micro = equilibrium(graph)
    assert np.allclose(equilibrium(macro), [micro[0], micro[1] + micro[2]])
    # Escape from the lumped states is slower than the microscopic step
    assert macro.rates[1, 0] < graph.rates[1, 0]


def test_coarse_graining_is_exact_when_lumpable() -> None:
    # Both members of the lump leave it at the same total rate
    graph = Graph(
        {
            ("a", "b"): 1.0,
            ("b", "a"): 1.0,
            ("b", "c"): 2.0,
            ("c", "b"): 1.0,
            ("a", "c"): 2.0,
            ("c", "a"): 1.0,
        }
    )
    groups = {"ab": ["a", "b"], "c": ["c"]}
    assert is_lumpable(graph, groups)
    macro = coarse_grain(graph, groups)
    assert np.allclose(macro.rates[0, 1], 2.0)
    assert np.allclose(
        coarse_grain(graph, groups).rates, passage_rates(graph, groups).rates
    )
    assert not is_lumpable(three_state_cycle(), groups)


def test_passage_rates_report_dwell_times() -> None:
    graph = landscape(
        {"free": 0.0, "search": -1.0, "bound": -3.0},
        {("free", "search"): 3.0, ("search", "bound"): 2.0},
    )
    groups = {"unbound": ["free"], "occupied": ["search", "bound"]}
    macro = passage_rates(graph, groups)
    dwell = mean_first_passage_times(graph, "free")
    local = equilibrium(graph)[1:] / equilibrium(graph)[1:].sum()
    assert np.isclose(macro.rates[1, 0], 1 / (local @ dwell[1:]))


def test_states_left_out_of_groups_stand_alone() -> None:
    graph = three_state_cycle()
    macro = coarse_grain(graph, {"ab": ["a", "b"]})
    assert macro.states == ("ab", "c")
    with pytest.raises(ValueError, match="two macrostates"):
        coarse_grain(graph, {"ab": ["a", "b"], "bc": ["b", "c"]})


def test_simulation_matches_the_master_equation() -> None:
    graph = three_state_cycle()
    times = np.array([0.0, 0.1, 0.3, 1.0, 5.0])
    molecules = 4000
    trajectory = simulate(graph, {"a": molecules}, times[-1], seed=0)
    occupancy = trajectory.occupancy(times) / molecules
    expected = propagate(graph, {"a": 1.0}, times)
    assert np.allclose(occupancy, expected, atol=4 / np.sqrt(molecules))
    assert np.allclose(occupancy.sum(axis=1), 1.0)


def test_simulation_of_a_single_molecule() -> None:
    graph = two_state()
    trajectory = simulate(graph, {"a": 1}, 50.0, seed=1)
    times, visited = trajectory.path(0)
    assert times[0] == 0.0 and visited[0] == 0
    assert np.all(np.diff(times) > 0)
    assert np.all(np.abs(np.diff(visited)) == 1)
    # Mean dwell times are the reciprocals of the rate constants
    dwell = np.diff(times)
    for state, rate in enumerate([3.0, 1.0]):
        held = dwell[visited[:-1] == state]
        assert np.isclose(held.mean(), 1 / rate, rtol=0.2)


def test_simulation_absorbs() -> None:
    graph = Graph({("a", "b"): 1.0})
    trajectory = simulate(graph, {"a": 20}, 10.0, seed=2)
    assert len(trajectory) == 20
    assert np.array_equal(trajectory.occupancy(10.0), [[0, 20]])
    assert np.array_equal(trajectory.occupancy(0.0), [[20, 0]])
    with pytest.raises(ValueError, match="nondecreasing"):
        trajectory.occupancy([1.0, 0.0])
    with pytest.raises(ValueError, match="whole numbers"):
        simulate(graph, {"a": 1.5}, 1.0)


def test_graph_is_differentiable() -> None:
    graph = three_state_cycle()

    def bound(log_rates: jax.Array) -> jax.Array:
        return equilibrium(graph.with_log_rates(log_rates))[2]

    gradient = jax.grad(bound)(graph.log_rates)
    assert gradient.shape == (3, 3)
    assert np.isfinite(
        gradient[np.isfinite(np.asarray(graph.log_rates))]
    ).all()
    step = 1e-6
    numerical = (
        bound(graph.log_rates.at[0, 1].add(step))
        - bound(graph.log_rates.at[0, 1].add(-step))
    ) / (2 * step)
    assert np.isclose(gradient[0, 1], numerical, rtol=1e-5)
    assert np.allclose(jax.jit(equilibrium)(graph), equilibrium(graph))


def test_state_diagram_draws_states_and_rates() -> None:
    graph = Graph({("a", "b"): 3.0, ("b", "a"): 1.0, ("b", "c"): 20.0})
    _, ax = plt.subplots()
    layout = state_diagram(graph, ax=ax)
    assert set(layout) == set(graph.states)
    labels = {text.get_text() for text in ax.texts}
    assert {"a", "b", "c"} <= labels
    assert {"3", "1", "20"} <= labels
    assert len(ax.patches) == len(graph.edges)
    points = np.stack([layout[state] for state in graph.states])
    for (low, high), coordinate in zip(
        [ax.get_xlim(), ax.get_ylim()], points.T
    ):  # whole nodes fit, rather than being clipped by the axis
        assert low < coordinate.min() and high > coordinate.max()

    _, other = plt.subplots()
    reused = state_diagram(graph, layout, rate_labels=False, ax=other)
    assert all(np.allclose(reused[s], layout[s]) for s in graph.states)
    assert {text.get_text() for text in other.texts} == {"a", "b", "c"}
    with pytest.raises(ValueError, match="no position for"):
        state_diagram(graph, {"a": (0.0, 0.0)}, ax=other)


def test_state_diagram_puts_the_forward_arrow_on_top() -> None:
    graph = Graph({("a", "b"): 2.0, ("b", "a"): 1.0})
    _, ax = plt.subplots()
    state_diagram(graph, {"a": (0.0, 0.0), "b": (1.0, 0.0)}, ax=ax)
    heights = {text.get_text(): text.get_position()[1] for text in ax.texts}
    # a -> b runs left to right, so its label sits above b -> a's
    assert heights["2"] > heights["1"]
    plt.close("all")


def test_energy_diagram_draws_states_and_barriers() -> None:
    graph = landscape(
        {"a": 0.0, "b": -2.0, "c": 1.0}, {("a", "b"): 4.0, ("b", "c"): 6.0}
    )
    _, ax = plt.subplots()
    line = energy_diagram(graph, reference="a", ax=ax)
    x, y = np.asarray(line.get_xydata()).T
    assert np.isclose(y.max(), 6.0)
    levels = [y[np.isclose(x, i - 0.25)][0] for i in range(3)]
    assert np.allclose(levels, [0.0, -2.0, 1.0])
    assert [text.get_text() for text in ax.get_xticklabels()] == [
        "a",
        "b",
        "c",
    ]
    with pytest.raises(ValueError, match="no transition"):
        energy_diagram(graph, ["a", "c"], ax=ax)


def recaptured() -> Graph:
    """Searching, testing and bound, with escapes usually recaptured."""
    return Graph(
        {
            ("searching", "testing"): 5.0,
            ("testing", "searching"): 2.0,
            ("testing", "bound"): 3.0,
            ("bound", "testing"): 0.4,
        }
    )


def stepped() -> Trajectory:
    """One molecule on a, b, c with a re-entry that must not count twice.

    Visits a at 0, b at 1, a at 2, b at 3, c at 5, a at 7, b at 8.
    """
    return Trajectory(
        states=("a", "b", "c"),
        initial=np.zeros(1, dtype=np.int64),
        times=np.array([1.0, 2.0, 3.0, 5.0, 7.0, 8.0]),
        molecules=np.zeros(6, dtype=np.int64),
        sources=np.array([0, 1, 0, 1, 2, 0]),
        targets=np.array([1, 0, 1, 2, 0, 1]),
    )


def test_dwell_rates_are_slower_when_escapes_are_recaptured() -> None:
    graph = recaptured()
    rates = dwell_rates(graph, "bound", "searching")
    assert rates["microscopic"] == pytest.approx(0.4)
    passage = mean_first_passage_times(graph, "searching")
    assert rates["observable"] == pytest.approx(
        1 / passage[graph.index("bound")]
    )
    # an escape that comes straight back lengthens the observed dwell
    assert rates["observable"] < rates["microscopic"]


def test_dwell_rates_without_a_target_give_only_the_microscopic_rate() -> None:
    assert dwell_rates(recaptured(), "bound") == {"microscopic": 0.4}


def test_dwells_per_visit_report_the_microscopic_rate() -> None:
    graph = recaptured()
    trace = simulate(graph, {"searching": 60}, 800.0, seed=0)
    dwells = trace.dwells("bound")
    expected = 1 / dwell_rates(graph, "bound")["microscopic"]
    assert dwells.mean() == pytest.approx(expected, rel=0.05)


def test_dwells_to_a_target_report_the_passage_time() -> None:
    graph = recaptured()
    trace = simulate(graph, {"searching": 60}, 800.0, seed=0)
    dwells = trace.dwells("bound", until="searching")
    expected = 1 / dwell_rates(graph, "bound", "searching")["observable"]
    assert dwells.mean() == pytest.approx(expected, rel=0.05)
    # the recaptured excursions make this the longer of the two
    assert dwells.mean() > trace.dwells("bound").mean()


def test_dwells_count_each_visit_once() -> None:
    trace = stepped()
    assert trace.dwells("a").tolist() == [1.0, 1.0, 1.0]
    # re-entering a before reaching c must not open a second dwell, and
    # the visit still running at the end is dropped
    assert trace.dwells("a", until="c").tolist() == [5.0]
    assert trace.dwells("b", until="a").tolist() == [1.0, 4.0]


def test_dwells_accept_indices_and_groups() -> None:
    trace = stepped()
    assert trace.dwells([0], until=[2]).tolist() == [5.0]
    assert trace.dwells("a", until=["b", "c"]).tolist() == [1.0, 1.0, 1.0]


def test_dwells_reject_a_target_that_is_also_the_state() -> None:
    with pytest.raises(ValueError, match="must not share"):
        stepped().dwells("a", until=["a", "c"])


def test_dwells_reject_an_unknown_state() -> None:
    with pytest.raises(KeyError, match="unknown state"):
        stepped().dwells("nowhere")


def test_dwells_of_a_state_never_visited_are_empty() -> None:
    graph = Graph({("a", "b"): 1.0, ("b", "a"): 1.0, ("b", "c"): 1.0})
    trace = simulate(graph, {"a": 1}, 0.0, seed=0)
    assert trace.dwells("c", until="a").size == 0


def test_dwell_diagram_draws_the_sample_and_its_rates() -> None:
    graph = recaptured()
    trace = simulate(graph, {"searching": 20}, 200.0, seed=0)
    dwells = trace.dwells("bound", until="searching")
    _, ax = plt.subplots()
    returned = dwell_diagram(graph, dwells, "bound", "searching", ax=ax)
    assert returned is ax
    labels = [line.get_label() for line in ax.lines]
    assert any("observable" in str(label) for label in labels)
    assert any("microscopic" in str(label) for label in labels)
    assert ax.patches  # the histogram
    plt.close("all")


def marklund() -> Scheme:
    """Searching, testing and bound, with every rate named."""
    return Scheme(
        {
            ("searching", "testing"): "k_on_max",
            ("testing", "searching"): "k_off_M",
            ("testing", "bound"): "k_on_mu",
            ("bound", "testing"): "k_off_mu",
        },
        latex={"k_off_mu": r"$k_\mathrm{off,\mu}$"},
    )


MARKLUND_RATES = {
    "k_on_max": 4.4e-3,
    "k_off_M": 100.0,
    "k_on_mu": 114.6,
    "k_off_mu": 6.0e-3,
}


def test_scheme_infers_its_states_and_transitions() -> None:
    scheme = marklund()
    assert scheme.states == ("searching", "testing", "bound")
    assert len(scheme) == 3
    assert scheme.index("bound") == 2
    assert ("bound", "testing") in scheme.edges
    assert len(scheme.edges) == 4
    assert "transitions=4" in repr(scheme)
    with pytest.raises(KeyError, match="unknown state"):
        scheme.index("nowhere")


def test_scheme_names_are_sorted_and_deduplicated() -> None:
    assert marklund().names == ("k_off_M", "k_off_mu", "k_on_max", "k_on_mu")
    shared = Scheme({("a", "b"): "k", ("b", "a"): "k"})
    assert shared.names == ("k",)


def test_scheme_labels_fall_back_to_the_name() -> None:
    scheme = marklund()
    assert scheme.label("k_off_mu") == r"$k_\mathrm{off,\mu}$"
    assert scheme.label("k_on_mu") == "k_on_mu"


def test_scheme_rejects_a_malformed_mechanism() -> None:
    with pytest.raises(ValueError, match="transition to themselves"):
        Scheme({("a", "a"): "k"})
    with pytest.raises(ValueError, match="latex names not in the scheme"):
        Scheme({("a", "b"): "k"}, latex={"typo": "K"})
    with pytest.raises(KeyError, match="unknown state"):
        Scheme({("a", "b"): "k"}, states=["a", "c"])
    with pytest.raises(ValueError, match="must be unique"):
        Scheme({("a", "b"): "k"}, states=["a", "a"])


def test_scheme_builds_a_graph_from_named_values() -> None:
    scheme = marklund()
    graph = scheme.graph(MARKLUND_RATES)
    built = Graph(
        {
            ("searching", "testing"): 4.4e-3,
            ("testing", "searching"): 100.0,
            ("testing", "bound"): 114.6,
            ("bound", "testing"): 6.0e-3,
        }
    )
    assert graph.states == scheme.states == built.states
    assert np.allclose(graph.log_rates, built.log_rates)
    assert np.allclose(equilibrium(graph), equilibrium(built))
    assert graph.temperature == scheme.temperature


def test_scheme_fills_every_edge_sharing_a_name() -> None:
    graph = Scheme({("a", "b"): "k", ("b", "a"): "k"}).graph({"k": 2.0})
    assert np.allclose(graph.rates[0, 1], graph.rates[1, 0])


def test_scheme_needs_a_value_for_every_name() -> None:
    scheme = marklund()
    with pytest.raises(ValueError, match=r"no value for \['k_off_mu'\]"):
        scheme.graph(
            {k: v for k, v in MARKLUND_RATES.items() if k != "k_off_mu"}
        )
    with pytest.raises(ValueError, match="not rate constants"):
        scheme.graph(MARKLUND_RATES | {"k_typo": 1.0})
    with pytest.raises(ValueError, match="must be positive"):
        scheme.graph(MARKLUND_RATES | {"k_off_mu": -1.0})


def test_scheme_overrides_one_rate_of_a_graph() -> None:
    scheme = marklund()
    wild = scheme.graph(MARKLUND_RATES)
    mutant = scheme.graph(wild, k_off_mu=6.0e-2)
    changed = ~np.isclose(
        np.asarray(wild.log_rates), np.asarray(mutant.log_rates)
    )
    assert changed.sum() == 1
    assert changed[wild.index("bound"), wild.index("testing")]
    assert np.isclose(mutant.rates[2, 1], 6.0e-2)
    # the mapping and the graph routes agree
    assert np.allclose(
        mutant.log_rates,
        scheme.graph(MARKLUND_RATES | {"k_off_mu": 6.0e-2}).log_rates,
    )


def test_scheme_rejects_a_graph_of_another_mechanism() -> None:
    scheme = marklund()
    with pytest.raises(ValueError, match="not of this mechanism"):
        scheme.graph(two_state(), k_off_mu=1.0)
    with pytest.raises(ValueError, match="not rate constants"):
        scheme.graph(scheme.graph(MARKLUND_RATES), k_typo=1.0)
    with pytest.raises(ValueError, match="must be positive"):
        scheme.graph(scheme.graph(MARKLUND_RATES), k_off_mu=0.0)


def test_scheme_override_keeps_a_graph_subclass() -> None:
    class Labelled(Graph):
        """A graph that carries the sequence its rates belong to."""

        sequence: str

    scheme = marklund()
    wild = Labelled(
        {edge: MARKLUND_RATES[name] for edge, name in scheme.symbols.items()},
        states=scheme.states,
    )
    wild.sequence = "CACGTG"
    mutant = scheme.graph(wild, k_off_mu=6.0e-2)
    assert isinstance(mutant, Labelled)
    assert mutant.sequence == "CACGTG"


def test_state_diagram_labels_a_scheme_with_its_names() -> None:
    scheme = marklund()
    _, ax = plt.subplots()
    layout = state_diagram(scheme, ax=ax)
    assert set(layout) == set(scheme.states)
    labels = {text.get_text() for text in ax.texts}
    assert set(scheme.states) <= labels
    # the named rate is drawn as its latex, the rest as they are spelled
    assert r"$k_\mathrm{off,\mu}$" in labels
    assert {"k_on_max", "k_off_M", "k_on_mu"} <= labels
    assert len(ax.patches) == len(scheme.edges)

    _, bare = plt.subplots()
    state_diagram(scheme, layout, rate_labels=False, ax=bare)
    assert {text.get_text() for text in bare.texts} == set(scheme.states)
    plt.close("all")


def test_residence_times_sum_to_the_passage_time() -> None:
    graph = recaptured()
    residence = mean_residence_times(graph, "searching")
    passage = mean_first_passage_times(graph, "searching")
    assert np.allclose(residence.sum(axis=1), passage)
    # the target neither holds time nor spends it
    index = graph.index("searching")
    assert np.allclose(residence[index], 0.0)
    assert np.allclose(residence[:, index], 0.0)


def test_residence_times_split_a_passage_time_by_state() -> None:
    # searching <- testing <-> bound, so a passage out of bound spends
    # 1/k_d bound and one testing lifetime per escape
    graph = recaptured()
    bound, testing = graph.index("bound"), graph.index("testing")
    residence = mean_residence_times(graph, "searching")
    p_tot = 3.0 / (3.0 + 2.0)
    assert residence[bound, bound] == pytest.approx(1 / (0.4 * (1 - p_tot)))
    assert residence[bound, testing] == pytest.approx(1 / 2.0)


def test_residence_times_give_the_number_of_visits() -> None:
    graph = recaptured()
    residence = mean_residence_times(graph, "searching")
    exits = -jnp.diag(graph.generator())
    visits = residence * exits
    bound = graph.index("bound")
    p_tot = 3.0 / (3.0 + 2.0)
    assert visits[bound, bound] == pytest.approx(1 / (1 - p_tot))


def test_residence_times_of_an_all_target_graph_are_zero() -> None:
    graph = two_state()
    assert np.allclose(mean_residence_times(graph, ["a", "b"]), 0.0)


def michaelis(substrate: float = 1.0, product: float = 1.0) -> Graph:
    """Enzyme turnover as a cycle, driven by the substrate excess.

    Both binding steps are pseudo-first-order, so a concentration
    multiplies the rate constant rather than appearing in the graph.
    """
    return Graph(
        {
            ("E", "ES"): 10.0 * substrate,
            ("ES", "E"): 1.0,
            ("ES", "EP"): 5.0,
            ("EP", "ES"): 2.0,
            ("EP", "E"): 20.0,
            ("E", "EP"): 4.0 * product,
        }
    )


def ruin(size: int = 5, hop: float = 2.0) -> Graph:
    """Symmetric hops along a line of states, s0 to s(size - 1)."""
    states = [f"s{i}" for i in range(size)]
    rates = {
        pair: hop
        for i in range(size - 1)
        for pair in ((states[i], states[i + 1]), (states[i + 1], states[i]))
    }
    return Graph(rates, states=states)


def test_fluxes_vanish_exactly_at_detailed_balance() -> None:
    assert np.allclose(fluxes(three_state_cycle()), 0, atol=1e-12)
    driven = fluxes(three_state_cycle(driven=True))
    assert not np.allclose(driven, 0)
    assert np.allclose(driven, -driven.T)


def test_fluxes_are_conserved_at_every_state() -> None:
    for graph in (three_state_cycle(driven=True), michaelis(3.0)):
        assert np.allclose(fluxes(graph).sum(axis=1), 0, atol=1e-12)


def test_flux_around_a_cycle_is_one_circulation() -> None:
    net = np.asarray(fluxes(three_state_cycle(driven=True)))
    assert np.allclose([net[0, 1], net[1, 2], net[2, 0]], net[0, 1])


def test_irreversible_flux_is_the_one_way_flux() -> None:
    graph = Graph({("a", "b"): 2.0, ("b", "c"): 3.0, ("c", "a"): 4.0})
    stationary = np.asarray(equilibrium(graph))
    assert np.isclose(np.asarray(fluxes(graph))[0, 1], stationary[0] * 2.0)


def test_cycle_affinity_measures_the_drive() -> None:
    assert np.isclose(
        cycle_affinity(three_state_cycle(), ["a", "b", "c"]), 0, atol=1e-12
    )
    driven = three_state_cycle(driven=True)
    affinity = cycle_affinity(driven, ["a", "b", "c"])
    assert affinity > 0
    # Reading the cycle backwards reverses the sign, as it does the flux
    assert np.isclose(cycle_affinity(driven, ["c", "b", "a"]), -affinity)
    assert np.sign(np.asarray(fluxes(driven))[0, 1]) == np.sign(affinity)


def test_cycle_affinity_is_infinite_without_a_reverse() -> None:
    graph = Graph({("a", "b"): 2.0, ("b", "c"): 3.0, ("c", "a"): 4.0})
    assert cycle_affinity(graph, ["a", "b", "c"]) == np.inf
    with pytest.raises(ValueError, match="no transition from 'a' to 'c'"):
        cycle_affinity(graph, ["a", "c", "b"])


def test_substrate_excess_sets_the_affinity_and_the_flux() -> None:
    cycle = ["E", "ES", "EP"]
    for substrate in (0.5, 2.0, 10.0):
        graph = michaelis(substrate)
        affinity = cycle_affinity(graph, cycle)
        assert np.isclose(
            affinity, RT * np.log(10.0 * substrate * 5.0 * 20.0 / 8.0)
        )
        assert np.sign(np.asarray(fluxes(graph))[1, 2]) == np.sign(affinity)
    # At the concentration where the drive vanishes the cycle runs at
    # equilibrium, so detailed balance is restored and turnover stops
    poised = michaelis(8.0 / (10.0 * 5.0 * 20.0))
    assert np.isclose(cycle_affinity(poised, cycle), 0, atol=1e-12)
    assert is_detailed_balanced(poised)
    assert np.allclose(fluxes(poised), 0, atol=1e-12)


def test_absorption_probabilities_split_a_branch_point() -> None:
    # Every path out of a passes through b, so both states report the
    # ratio of the two rates leaving b
    graph = Graph({("a", "b"): 7.0, ("b", "c"): 3.0, ("b", "d"): 1.0})
    reached = np.asarray(absorption_probabilities(graph, ["c", "d"]))
    assert np.allclose(reached.sum(axis=1), 1)
    for state in ("a", "b"):
        assert np.allclose(reached[graph.index(state)], [0, 0, 0.75, 0.25])
    assert np.allclose(reached[graph.index("c")], [0, 0, 1, 0])


def test_absorption_probabilities_solve_a_gamblers_ruin() -> None:
    # Symmetric hops on a line reach the far end with a probability
    # linear in the starting position, whatever the hop rate
    graph = ruin()
    reached = np.asarray(absorption_probabilities(graph, ["s0", "s4"]))
    assert np.allclose(reached.sum(axis=1), 1)
    assert np.allclose(
        reached[:, graph.index("s4")], [0, 0.25, 0.5, 0.75, 1.0]
    )


def test_absorption_probabilities_follow_from_the_residence_times() -> None:
    # Absorption is one step out of the last free state visited, so the
    # two solves are related by the rates into the targets
    graph, targets = ruin(), ["s0", "s4"]
    free = [graph.index(state) for state in ("s1", "s2", "s3")]
    hit = [graph.index(state) for state in targets]
    residence = np.asarray(mean_residence_times(graph, targets))
    predicted = (residence @ np.asarray(graph.rates))[np.ix_(free, hit)]
    reached = np.asarray(absorption_probabilities(graph, targets))
    assert np.allclose(reached[np.ix_(free, hit)], predicted)


def test_absorption_probabilities_match_a_simulation() -> None:
    graph = Graph({("b", "a"): 1.0, ("b", "c"): 3.0})
    reached = np.asarray(absorption_probabilities(graph, ["a", "c"]))
    assert np.isclose(reached[graph.index("b"), graph.index("c")], 0.75)
    trace = simulate(graph, {"b": 4000}, duration=50.0, seed=0)
    assert np.isclose(
        (trace.targets == graph.index("c")).mean(), 0.75, atol=0.02
    )


def test_absorption_probabilities_of_degenerate_targets() -> None:
    assert np.allclose(
        absorption_probabilities(two_state(), ["a", "b"]), np.eye(2)
    )
    with pytest.raises(ValueError, match="at least one target"):
        absorption_probabilities(two_state(), [])


def test_state_diagram_leaves_room_for_curved_arrows() -> None:
    # The arrow between the far apart states bows well outside the box
    # the three node centers span, and must not be clipped
    graph = Graph(
        {("a", "b"): 1.0, ("b", "a"): 1.0, ("a", "c"): 1.0, ("c", "a"): 1.0}
    )
    pos = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (2.0, 0.0)}
    _, ax = plt.subplots()
    state_diagram(graph, pos, ax=ax, curvature=0.5, rate_labels=False)
    # arc3 reaches half its control offset, so a span of 2 at rad 0.5
    # bows 0.5 out of a layout that is flat in y
    assert ax.get_ylim()[1] >= 0.5
    assert ax.get_ylim()[0] <= -0.5
    _, straight = plt.subplots()
    state_diagram(graph, pos, ax=straight, curvature=0.0, rate_labels=False)
    assert straight.get_ylim()[1] < ax.get_ylim()[1]
    plt.close("all")


def test_state_diagram_colors_transitions_in_edge_order() -> None:
    # A scheme orders its transitions as written, which need not be the
    # order the drawing library would pick, so a per-transition color
    # has to be matched back to the transition it was meant for. Each
    # transition here leaves a different state, so where an arrow starts
    # says which one it is.
    scheme = Scheme(
        {("b", "c"): "second", ("a", "b"): "first", ("c", "a"): "third"}
    )
    assert scheme.edges == (("b", "c"), ("a", "b"), ("c", "a"))
    pos = {
        state: np.array(point)
        for state, point in (
            ("a", (0.0, 0.0)),
            ("b", (10.0, 0.0)),
            ("c", (5.0, 9.0)),
        )
    }
    wanted = {"b": "red", "a": "green", "c": "blue"}
    _, ax = plt.subplots()
    state_diagram(
        scheme,
        pos,
        ax=ax,
        curvature=0.0,
        rate_labels=False,
        edge_color=[wanted[source] for source, _ in scheme.edges],
    )

    drawn: dict[str, str] = {}
    for patch in ax.patches:
        if not isinstance(patch, matplotlib.patches.FancyArrowPatch):
            continue
        start = np.asarray(patch.get_path().vertices)[0]
        source = min(
            pos, key=lambda state: float(np.hypot(*(pos[state] - start)))
        )
        drawn[source] = matplotlib.colors.to_hex(patch.get_edgecolor())
    assert drawn == {
        source: matplotlib.colors.to_hex(color)
        for source, color in wanted.items()
    }
    plt.close("all")
