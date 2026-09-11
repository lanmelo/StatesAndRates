"""Observable rates of a graph whose states are partly unresolved.

Microscopic states are often not distinguishable in an experiment, so a
measured rate constant is some combination of the microscopic ones. Both
coarse grainings here lump the states into macrostates and return a
graph over those macrostates, which every other analysis then accepts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy import typing as npt

from .graph import Graph
from .thermo import equilibrium

Groups = Mapping[str, Sequence[str]]
Indices = npt.NDArray[np.int64]
Selection = str | Sequence[str] | Sequence[int] | Indices


def coarse_grain(graph: Graph, groups: Groups) -> Graph:
    """Lump states, averaging rates over the local equilibrium.

    The macro rate K_AB is the equilibrium average over the states of A
    of the total rate into the states of B. This reproduces the
    equilibrium populations and the equilibrium fluxes between
    macrostates exactly, and is a Markovian approximation of the
    observable kinetics that is exact when the graph is lumpable. It
    inherits detailed balance from the graph it coarse grains.

    Args:
        graph: Graph to coarse grain.
        groups: States making up each macrostate, keyed by its name.
            States left out form their own macrostate.

    Returns:
        A graph over the macrostates.
    """
    names, members = _partition(graph, groups)
    weights = _local_equilibrium(graph, members)
    total = jnp.stack(
        [graph.rates[:, block].sum(axis=1) for block in members], axis=1
    )
    rates = jnp.stack(
        [weight @ total[block] for weight, block in zip(weights, members)]
    )
    return _macro_graph(graph, names, rates)


def passage_rates(graph: Graph, groups: Groups) -> Graph:
    """Lump states, taking rates from mean first passage times.

    The macro rate K_AB is the reciprocal of the equilibrium average
    over the states of A of the mean time to first reach B. This is the
    rate that a measured mean dwell time reports, and each pair of
    macrostates is treated on its own, so the result reproduces dwell
    times rather than equilibrium populations.

    Args:
        graph: Graph to coarse grain; must be irreducible.
        groups: States making up each macrostate, keyed by its name.
            States left out form their own macrostate.

    Returns:
        A graph over the macrostates.
    """
    names, members = _partition(graph, groups)
    weights = _local_equilibrium(graph, members)
    passage = jnp.stack(
        [mean_first_passage_times(graph, block) for block in members], axis=1
    )
    rates = 1 / jnp.stack(
        [weight @ passage[block] for weight, block in zip(weights, members)]
    )
    return _macro_graph(graph, names, rates)


def mean_first_passage_times(graph: Graph, target: Selection) -> Array:
    """Mean time to first reach a target state from every state.

    Solves sum_j Q_ij m_j = -1 over the states outside the target, where
    m is zero. Target states themselves give zero. Equivalently the row
    sums of `mean_residence_times`, but as a single right-hand side.

    Args:
        graph: Graph to solve; every state must reach the target.
        target: Name or names of the target states, or their indices.
    """
    absorbed = np.zeros(len(graph), dtype=bool)
    absorbed[_indices(graph, target)] = True
    free = np.flatnonzero(~absorbed)
    if not free.size:
        return jnp.zeros(len(graph))
    generator = graph.generator()[np.ix_(free, free)]
    times = jnp.linalg.solve(generator, -jnp.ones(len(free)))
    return jnp.zeros(len(graph)).at[free].set(times)


def mean_residence_times(graph: Graph, target: Selection) -> Array:
    """Mean time spent in each state before first reaching a target.

    Entry (i, j) is the total time spent in j, summed over every visit,
    starting from i and stopping on arrival in the target. Rows and
    columns of target states are zero.

    Solves Q_FF X = -I over the states outside the target, so it is the
    matrix whose row sums are `mean_first_passage_times`. Two readings
    of it are worth knowing:

    * dividing entry (i, j) by the mean time per visit to j, which is
      one over the total rate out of j, gives the expected *number* of
      visits to j. For a state that is left and re-entered, that is how
      often an escape is recaptured.
    * splitting a passage time across states says where the time goes,
      which a single passage time cannot.

    Costs one solve with n right-hand sides rather than one, so prefer
    `mean_first_passage_times` when only the totals are wanted.

    Args:
        graph: Graph to solve; every state must reach the target.
        target: Name or names of the target states, or their indices.
    """
    absorbed = np.zeros(len(graph), dtype=bool)
    absorbed[_indices(graph, target)] = True
    free = np.flatnonzero(~absorbed)
    if not free.size:
        return jnp.zeros((len(graph),) * 2)
    generator = graph.generator()[np.ix_(free, free)]
    residence = jnp.linalg.solve(generator, -jnp.eye(len(free)))
    return jnp.zeros((len(graph),) * 2).at[np.ix_(free, free)].set(residence)


def absorption_probabilities(graph: Graph, targets: Selection) -> Array:
    """Which target is reached first, from every state.

    Entry (i, j) is the probability that j is the first target state
    reached, starting from i; a target reaches itself, so its row is one
    in its own column. Rows sum to one whenever every state can reach
    some target.

    Solves Q_FF H = -Q_FA over the states outside the target set, the
    same matrix `mean_residence_times` inverts, with the transitions
    into the targets as the right-hand side. Where a passage time says
    how long a fate takes, this says which fate it is, so it is the
    quantity to ask for at a branch point: the chance that a bound
    molecule goes on rather than back.

    Args:
        graph: Graph to solve; every state must reach a target.
        targets: Names of the target states, or their indices. Two or
            more make the answer more than a column of ones.
    """
    absorbed = np.zeros(len(graph), dtype=bool)
    hit = _indices(graph, targets)
    if not hit.size:
        raise ValueError("at least one target is required")
    absorbed[hit] = True
    free = np.flatnonzero(~absorbed)
    reached = jnp.zeros((len(graph),) * 2).at[hit, hit].set(1.0)
    if not free.size:
        return reached
    generator = graph.generator()
    solved = jnp.linalg.solve(
        generator[np.ix_(free, free)], -generator[np.ix_(free, hit)]
    )
    return reached.at[np.ix_(free, hit)].set(solved)


def is_lumpable(graph: Graph, groups: Groups, *, rtol: float = 1e-8) -> bool:
    """Whether lumping the states leaves the kinetics Markovian.

    A partition is lumpable when every state of a macrostate has the
    same total rate into each other macrostate, in which case the macro
    rates do not depend on how the states within a macrostate are
    occupied. This inspects rate constants, so it needs concrete values.

    Args:
        graph: Graph to test.
        groups: States making up each macrostate, keyed by its name.
        rtol: Tolerance on the spread of rates within a macrostate,
            relative to their mean.
    """
    _, members = _partition(graph, groups)
    rates = np.asarray(graph.rates)
    for block in members:
        for other in members:
            if other is block:
                continue
            out = rates[np.ix_(block, other)].sum(axis=1)
            if not np.allclose(out, out.mean(), rtol=rtol, atol=0):
                return False
    return True


def dwell_rates(
    graph: Graph, state: str | int, until: Selection | None = None
) -> dict[str, float]:
    """The two rates a dwell in one state can report.

    The microscopic rate is the total rate out of the state, so the time
    per visit is exponential with it. The observable rate is the
    reciprocal of the mean first passage time to the target, which is
    what a measurement reports when the excursions that come straight
    back are not resolved, and is the slower of the two. They are the
    expectations for Trajectory.dwells without and with a target.

    Args:
        graph: Graph to read the rates from.
        state: State the dwell is in.
        until: State that ends an observable dwell, or several. Left
            out, only the microscopic rate is returned.

    Returns:
        The rates, keyed by name.
    """
    index = graph.index(state) if isinstance(state, str) else int(state)
    rates = {"microscopic": float(graph.rates[index].sum())}
    if until is not None:
        passage = mean_first_passage_times(graph, until)
        rates["observable"] = float(1.0 / passage[index])
    return rates


def _indices(graph: Graph, states: Selection) -> Indices:
    """State indices from names, indices, or a single name."""
    if isinstance(states, str):
        states = [states]
    return np.array(
        [
            graph.index(state) if isinstance(state, str) else int(state)
            for state in states
        ],
        dtype=np.int64,
    )


def _partition(
    graph: Graph, groups: Groups
) -> tuple[tuple[str, ...], list[Indices]]:
    """Macrostate names and the state indices each one contains."""
    members = [_indices(graph, group) for group in groups.values()]
    assigned = np.concatenate(members) if members else np.zeros(0, np.int64)
    if len(np.unique(assigned)) != len(assigned):
        raise ValueError("states cannot belong to two macrostates")
    names = list(groups)
    for i in range(len(graph)):
        if i not in assigned:
            names.append(graph.states[i])
            members.append(np.array([i], dtype=np.int64))
    return tuple(names), members


def _local_equilibrium(graph: Graph, members: list[Indices]) -> list[Array]:
    """Equilibrium occupancy within each macrostate, summing to one."""
    stationary = equilibrium(graph)
    return [stationary[block] / stationary[block].sum() for block in members]


def _macro_graph(graph: Graph, names: tuple[str, ...], rates: Array) -> Graph:
    """Graph over macrostates, dropping self transitions."""
    log_rates = jnp.log(rates).at[jnp.diag_indices(len(names))].set(-jnp.inf)
    return Graph(
        log_rates=log_rates, states=names, temperature=graph.temperature
    )
