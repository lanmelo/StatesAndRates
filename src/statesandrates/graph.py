"""Graphs of states connected by first-order rate constants."""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from typing import Self, override

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants
from jax import Array
from numpy import typing as npt

Edge = tuple[str, str]
Floats = npt.NDArray[np.float64]
Amounts = Mapping[str, float] | npt.ArrayLike

GAS_KCAL_MOL_K: float = scipy.constants.gas_constant / (
    scipy.constants.kilo * scipy.constants.calorie
)


def thermal_energy(temperature: float = 298.15) -> float:
    """Thermal energy RT in kcal/mol at a temperature in kelvin."""
    return GAS_KCAL_MOL_K * temperature


class Graph:
    """States connected by first-order rate constants.

    Rate constants are held as logarithms, so that mutations shift them
    additively and a transition absent from the graph is -inf.

    A graph is a JAX pytree whose only leaf is its log rate constants, so
    it can be passed through jit, grad and vmap directly. Its state names
    and temperature are static.

    Args:
        rates: Rate constant of each i -> j transition, keyed by the
            names of the two states.
        log_rates: Log rate constants, either keyed by transition or as a
            full (n, n) matrix using -inf for absent transitions.
            Mutually exclusive with rates.
        states: Names of the states in the order they are indexed;
            inferred from the transitions if not given, and required when
            log_rates is a matrix.
        temperature: Temperature in kelvin, used to convert rate
            constants into free energies.

    Attributes:
        states: Names of the states in the order they are indexed.
        log_rates: Matrix of log rate constants, where entry (i, j) is
            the i -> j transition and -inf marks an absent transition.
        temperature: Temperature in kelvin.
    """

    states: tuple[str, ...]
    log_rates: Array
    temperature: float

    def __init__(
        self,
        rates: Mapping[Edge, float] | None = None,
        *,
        log_rates: Mapping[Edge, float] | npt.ArrayLike | None = None,
        states: Sequence[str] | None = None,
        temperature: float = 298.15,
    ) -> None:
        if (rates is None) == (log_rates is None):
            raise ValueError("specify exactly one of rates, log_rates")
        if rates is not None:
            if not all(rate > 0 for rate in rates.values()):
                raise ValueError("rate constants must be positive")
            log_rates = {
                edge: float(np.log(rate)) for edge, rate in rates.items()
            }

        matrix: Floats
        if isinstance(log_rates, Mapping):
            if states is None:
                states = list(
                    dict.fromkeys(name for edge in log_rates for name in edge)
                )
            names = tuple(states)
            matrix = np.full((len(names), len(names)), -np.inf)
            for (source, target), log_rate in log_rates.items():
                matrix[names.index(source), names.index(target)] = log_rate
        else:
            if states is None:
                raise ValueError("states is required for a matrix of rates")
            names = tuple(states)
            matrix = np.array(log_rates, dtype=float)

        if len(set(names)) != len(names):
            raise ValueError("state names must be unique")
        if matrix.shape != (len(names),) * 2:
            raise ValueError(f"log_rates must have shape {(len(names),) * 2}")
        if np.isnan(matrix).any() or np.isposinf(matrix).any():
            raise ValueError("log rate constants must be finite or -inf")
        if np.isfinite(matrix.diagonal()).any():
            raise ValueError("states cannot transition to themselves")
        self.states = names
        self.log_rates = jnp.asarray(matrix)
        self.temperature = temperature

    def with_log_rates(self, log_rates: Array) -> Self:
        """Copy of the graph with new log rates and the same states.

        The new rates are not validated, so this is safe to call on
        traced values. Whatever a subclass adds is carried over.
        """
        graph = copy(self)
        graph.log_rates = log_rates
        return graph

    def __len__(self) -> int:
        return len(self.states)

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(states={self.states},"
            f" transitions={len(self.edges)})"
        )

    @property
    def rates(self) -> Array:
        """Matrix of rate constants, zero for absent transitions."""
        return jnp.exp(self.log_rates)

    @property
    def edges(self) -> tuple[Edge, ...]:
        """Transitions present in the graph."""
        rows, columns = np.nonzero(np.isfinite(np.asarray(self.log_rates)))
        return tuple(
            (self.states[i], self.states[j]) for i, j in zip(rows, columns)
        )

    @property
    def rt(self) -> float:
        """Thermal energy in kcal/mol."""
        return thermal_energy(self.temperature)

    def index(self, state: str) -> int:
        """Position of a state in the state ordering."""
        try:
            return self.states.index(state)
        except ValueError:
            raise KeyError(f"unknown state {state!r}") from None

    def vector(self, amounts: Amounts) -> Array:
        """Amount in each state, from a mapping or an array.

        States missing from a mapping are zero.
        """
        if isinstance(amounts, Mapping):
            dense = np.zeros(len(self))
            for state, amount in amounts.items():
                dense[self.index(state)] = amount
            return jnp.asarray(dense)
        vector = jnp.asarray(amounts, dtype=float)
        if vector.shape != (len(self),):
            raise ValueError(f"expected an amount for {len(self)} states")
        return vector

    def generator(self) -> Array:
        """Rate matrix Q of the master equation dp/dt = Q.T @ p."""
        rates = self.rates
        return rates - jnp.diag(rates.sum(axis=1))

    def mutate(self, *mutations: Mutation) -> Self:
        """Copy of the graph with its rate constants perturbed."""
        log_rates = self.log_rates
        for mutation in mutations:
            log_rates = log_rates + mutation.log_shift(self)
        return self.with_log_rates(log_rates)


def _new(
    states: tuple[str, ...], log_rates: Array, temperature: float
) -> Graph:
    """A graph whose rate constants are taken as they are given."""
    graph = object.__new__(Graph)
    graph.states = states
    graph.log_rates = log_rates
    graph.temperature = temperature
    return graph


jax.tree_util.register_pytree_node(
    Graph,
    lambda graph: ((graph.log_rates,), (graph.states, graph.temperature)),
    lambda static, leaves: _new(static[0], leaves[0], static[1]),
)


@dataclass(frozen=True)
class Mutation:
    """A perturbation of the free energy landscape of a graph.

    Energies are in kcal/mol and positive values are destabilizing.
    Raising the free energy of a state speeds up every transition out of
    it, and raising the barrier of a transition slows down both of its
    directions, so either perturbation leaves the graph thermodynamically
    consistent. Rate factors act on a single direction and are
    unconstrained. Perturbing an absent transition has no effect.

    Mutations add together with ``+``, and are applied by
    ``Graph.mutate``.

    Attributes:
        states: Change in the free energy of a state.
        barriers: Change in the barrier height of a transition, keyed by
            either of its two directions.
        factors: Factor multiplying the rate constant of a transition.
        name: Optional label.
    """

    states: Mapping[str, float] = field(default_factory=dict)
    barriers: Mapping[Edge, float] = field(default_factory=dict)
    factors: Mapping[Edge, float] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        barriers: dict[Edge, float] = {}
        for (source, target), energy in self.barriers.items():
            key = (min(source, target), max(source, target))
            barriers[key] = barriers.get(key, 0.0) + energy
        object.__setattr__(self, "barriers", barriers)

    def __add__(self, other: Mutation) -> Self:
        return replace(
            self,
            states=_merge(self.states, other.states),
            barriers=_merge(self.barriers, other.barriers),
            factors=_merge(self.factors, other.factors, operator.mul),
            name=" + ".join(name for name in (self.name, other.name) if name),
        )

    def log_shift(self, graph: Graph) -> Floats:
        """Shift this mutation applies to the log rates of a graph."""
        shift = np.zeros((len(graph), len(graph)))
        for state, energy in self.states.items():
            shift[graph.index(state)] += energy / graph.rt
        for (source, target), energy in self.barriers.items():
            i, j = graph.index(source), graph.index(target)
            shift[[i, j], [j, i]] -= energy / graph.rt
        for (source, target), factor in self.factors.items():
            if factor <= 0:
                raise ValueError("rate factors must be positive")
            shift[graph.index(source), graph.index(target)] += np.log(factor)
        return shift


def _merge[KeyT](
    left: Mapping[KeyT, float],
    right: Mapping[KeyT, float],
    combine: Callable[[float, float], float] = operator.add,
) -> dict[KeyT, float]:
    """Union of two mappings, combining the shared keys."""
    merged = dict(left)
    for key, value in right.items():
        merged[key] = combine(merged[key], value) if key in merged else value
    return merged
