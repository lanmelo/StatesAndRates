"""Stationary distributions, free energies and fluxes of a graph."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from .graph import Graph


def equilibrium(graph: Graph) -> Array:
    """Normalized stationary distribution over states.

    The graph must be irreducible, so that the distribution is unique.
    """
    return _gth(graph.generator())


def log_equilibrium(graph: Graph) -> Array:
    """Log of the normalized stationary distribution over states."""
    return jnp.log(equilibrium(graph))


def free_energies(graph: Graph, *, reference: str | None = None) -> Array:
    """Free energy of each state in kcal/mol.

    Args:
        graph: Graph to solve.
        reference: State taken as the zero of energy; by default the
            energies are -RT log of the normalized stationary
            distribution.
    """
    energies = -graph.rt * log_equilibrium(graph)
    if reference is not None:
        energies = energies - energies[graph.index(reference)]
    return energies


def barrier_energies(
    graph: Graph, *, reference: str | None = None, prefactor: float = 1.0
) -> Array:
    """Free energy of each transition state in kcal/mol.

    Entry (i, j) is the barrier separating states i and j, or nan if
    they do not interconvert. Detailed balance is required, without which
    the two directions of a transition imply different barriers.

    Args:
        graph: Graph to solve.
        reference: State taken as the zero of energy.
        prefactor: Rate constant of a barrierless transition, in the
            units of the graph. It shifts every barrier by RT log of
            itself and so leaves their differences unchanged.
    """
    if not is_detailed_balanced(graph):
        raise ValueError("barrier heights require detailed balance")
    energies = free_energies(graph, reference=reference)
    barriers = energies[:, None] - graph.rt * (
        graph.log_rates - float(np.log(prefactor))
    )
    return jnp.asarray(
        jnp.where(jnp.isfinite(graph.log_rates), barriers, jnp.nan)
    )


def fluxes(graph: Graph) -> Array:
    """Net flux through each transition in the stationary state.

    Entry (i, j) is pi_i k_ij - pi_j k_ji, the net rate at which
    probability crosses the transition. The matrix is antisymmetric and
    its rows sum to zero, which is stationarity written one state at a
    time. Every entry is zero exactly when the graph is detailed
    balanced, so a nonzero flux is circulation driven by whatever breaks
    it, and `cycle_affinity` measures the force behind it.

    A transition with no reverse carries all of its one-way flux, which
    is the turnover of an irreversible step. One-way fluxes in general
    are `equilibrium(graph)[:, None] * graph.rates`.
    """
    one_way = equilibrium(graph)[:, None] * graph.rates
    return one_way - one_way.T


def cycle_affinity(graph: Graph, cycle: Sequence[str]) -> float:
    """Free energy dissipated per turn of a cycle, in kcal/mol.

    RT times the log of the product of the forward rate constants around
    the cycle over the product of the reverse ones. Kolmogorov's
    criterion is that this vanishes for every cycle, so it is the
    quantity `is_detailed_balanced` tests, and it is the thermodynamic
    force driving the flux `fluxes` reports. A transition whose reverse
    is absent gives an infinite affinity, a one-way step being an
    infinite driving force.

    This inspects rate constants, so it needs concrete values.

    Args:
        graph: Graph to measure.
        cycle: States in the order they are visited, closing from the
            last back to the first, which is not repeated.
    """
    log_rates = np.asarray(graph.log_rates)
    steps = [graph.index(state) for state in cycle]
    total = 0.0
    for step, (i, j) in enumerate(zip(steps, steps[1:] + steps[:1])):
        if not np.isfinite(log_rates[i, j]):
            raise ValueError(
                f"no transition from {cycle[step]!r}"
                f" to {cycle[(step + 1) % len(cycle)]!r}"
            )
        total += log_rates[i, j] - log_rates[j, i]
    return graph.rt * total


def is_detailed_balanced(graph: Graph, *, atol: float = 1e-8) -> bool:
    """Whether the graph is reversible and satisfies detailed balance.

    Detailed balance holds when every transition has a reverse and
    log(k_ij / k_ji) sums to zero around every cycle, the Kolmogorov
    condition. It is tested by summing log(k_ij / k_ji) along a spanning
    tree of the graph, which fixes the equilibrium up to a constant, and
    taking the residual over the remaining transitions.

    This inspects rate constants, so it needs concrete values.

    Args:
        graph: Graph to test; must be connected.
        atol: Tolerance on the residual, in log rate constants.
    """
    log_rates = np.asarray(graph.log_rates)
    present = np.isfinite(log_rates)
    if not np.array_equal(present, present.T):
        return False
    with np.errstate(invalid="ignore"):  # -inf - -inf, masked out below
        log_ratio = log_rates - log_rates.T
    log_p = np.full(len(graph), np.nan)
    log_p[0] = 0.0
    stack = [0]
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(present[i] & np.isnan(log_p)):
            log_p[j] = log_p[i] + log_ratio[i, j]
            stack.append(int(j))
    if np.isnan(log_p).any():
        raise ValueError("graph is not connected")
    residual = log_p[None, :] - log_p[:, None] - log_ratio
    return bool(np.all(np.abs(residual[present]) <= atol))


def _gth(generator: Array) -> Array:
    """Stationary distribution by Grassmann-Taksar-Heyman reduction.

    Eliminates one state at a time using only the nonnegative
    off-diagonal rates, so no cancellation can occur. See Grassmann,
    Taksar and Heyman (1985), Operations Research 33, 1107.
    """
    matrix = generator
    for k in range(len(matrix) - 1, 0, -1):
        matrix = matrix.at[:k, k].divide(matrix[k, :k].sum())
        matrix = matrix.at[:k, :k].add(jnp.outer(matrix[:k, k], matrix[k, :k]))
    weights = jnp.ones(len(matrix))
    for k in range(1, len(matrix)):
        weights = weights.at[k].set(weights[:k] @ matrix[:k, k])
    return weights / weights.sum()
