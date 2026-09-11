"""Mechanisms written as named rate constants, with no values attached."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import override

import jax.numpy as jnp
import numpy as np

from .graph import Edge, Graph

Symbols = Mapping[Edge, str]
Values = Mapping[str, float] | Graph


class Scheme:
    """States connected by named rate constants.

    A scheme is a mechanism without values: every transition carries the
    name of its rate constant rather than a number. Supplying a value for
    each name gives an ordinary graph, which is what the rest of the
    package analyses. A scheme holds no rate constants of its own and is
    not a JAX pytree, so it cannot be handed to one of those analyses by
    mistake.

    Args:
        symbols: Name of the rate constant of each i -> j transition,
            keyed by the names of the two states. A name repeated on
            several transitions gives all of them the same value.
        latex: How to draw each name in a diagram, keyed by the name.
            Names left out are drawn as they are spelled.
        states: Names of the states in the order they are indexed;
            inferred from the transitions if not given.
        temperature: Temperature in kelvin, passed to the graphs this
            scheme builds.

    Attributes:
        states: Names of the states in the order they are indexed.
        symbols: Name of the rate constant of each transition.
        latex: How to draw each name in a diagram.
        temperature: Temperature in kelvin.
    """

    states: tuple[str, ...]
    symbols: Symbols
    latex: Mapping[str, str]
    temperature: float

    def __init__(
        self,
        symbols: Symbols,
        *,
        latex: Mapping[str, str] | None = None,
        states: Sequence[str] | None = None,
        temperature: float = 298.15,
    ) -> None:
        if states is None:
            states = list(
                dict.fromkeys(state for edge in symbols for state in edge)
            )
        names = tuple(states)
        if len(set(names)) != len(names):
            raise ValueError("state names must be unique")
        for source, target in symbols:
            if source == target:
                raise ValueError("states cannot transition to themselves")
            for state in (source, target):
                if state not in names:
                    raise KeyError(f"unknown state {state!r}")
        stray = set(latex or ()) - set(symbols.values())
        if stray:
            raise ValueError(f"latex names not in the scheme: {sorted(stray)}")
        self.states = names
        self.symbols = dict(symbols)
        self.latex = dict(latex or {})
        self.temperature = temperature

    def __len__(self) -> int:
        return len(self.states)

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(states={self.states},"
            f" transitions={len(self.edges)})"
        )

    @property
    def edges(self) -> tuple[Edge, ...]:
        """Transitions present in the scheme."""
        return tuple(self.symbols)

    @property
    def names(self) -> tuple[str, ...]:
        """Rate constant names, each of which needs a value."""
        return tuple(sorted(set(self.symbols.values())))

    def index(self, state: str) -> int:
        """Position of a state in the state ordering."""
        try:
            return self.states.index(state)
        except ValueError:
            raise KeyError(f"unknown state {state!r}") from None

    def label(self, name: str) -> str:
        """How a rate constant name is drawn in a diagram."""
        return self.latex.get(name, name)

    def graph(self, base: Values, /, **overrides: float) -> Graph:
        """Graph with a value for every rate constant.

        Args:
            base: A value for each name, or a graph this scheme built
                already, whose rates are kept except where overridden.
            **overrides: Values replacing those of base, by name.

        Returns:
            A graph over the same states in the same order.

        Raises:
            ValueError: If a name is not one of the scheme's, if one is
                left without a value, if a value is not positive, or if
                base is a graph of a different mechanism.
        """
        if isinstance(base, Graph):
            self._known(overrides)
            return self._replace(base, overrides)
        values = {**base, **overrides}
        self._known(values)
        missing = set(self.names) - set(values)
        if missing:
            raise ValueError(f"no value for {sorted(missing)}")
        return Graph(
            {edge: values[name] for edge, name in self.symbols.items()},
            states=self.states,
            temperature=self.temperature,
        )

    def _known(self, values: Mapping[str, float]) -> None:
        """Reject names the scheme does not have."""
        stray = set(values) - set(self.names)
        if stray:
            raise ValueError(
                f"not rate constants of this scheme: {sorted(stray)}"
            )

    def _replace(self, base: Graph, values: Mapping[str, float]) -> Graph:
        """Copy of a graph with some of its named rates given new values."""
        if base.states != self.states or set(base.edges) != set(self.edges):
            raise ValueError("graph is not of this mechanism")
        if not all(value > 0 for value in values.values()):
            raise ValueError("rate constants must be positive")
        log_rates = np.asarray(base.log_rates).copy()
        for (source, target), name in self.symbols.items():
            if name in values:
                row, column = self.index(source), self.index(target)
                log_rates[row, column] = float(np.log(values[name]))
        return base.with_log_rates(jnp.asarray(log_rates))
