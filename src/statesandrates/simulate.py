"""Gillespie simulation of individual molecules on a graph.

Every transition is first order, so molecules never interact and each
one is an independent continuous-time Markov jump process. They are
simulated together by the first reaction method, which draws a waiting
time for every transition out of the current state and takes the
shortest. Waiting times are drawn in log space, so rate constants
spanning many orders of magnitude are handled without loss of accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import numpy as np
from numpy import typing as npt

from .graph import Amounts, Graph

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]
Selection = str | Sequence[str] | Sequence[int] | Ints


@dataclass(frozen=True)
class Trajectory:
    """Jump events of independent molecules, ordered by time.

    Attributes:
        states: Names of the states, indexed by the event arrays.
        initial: State of each molecule at time zero.
        times: Time of each jump.
        molecules: Molecule that jumped.
        sources: State jumped from.
        targets: State jumped to.
    """

    states: tuple[str, ...]
    initial: Ints
    times: Floats
    molecules: Ints
    sources: Ints
    targets: Ints

    def __len__(self) -> int:
        return len(self.times)

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(molecules={len(self.initial)},"
            f" jumps={len(self)})"
        )

    @property
    def counts(self) -> Ints:
        """Number of molecules in each state at time zero."""
        counts = np.bincount(self.initial, minlength=len(self.states))
        return np.asarray(counts, dtype=np.int64)

    def occupancy(self, times: npt.ArrayLike) -> Ints:
        """Number of molecules in each state at the given times.

        Args:
            times: Nondecreasing times at which to count molecules.

        Returns:
            Counts, of shape (len(times), len(states)).
        """
        time = np.atleast_1d(np.asarray(times, dtype=float))
        if (np.diff(time) < 0).any():
            raise ValueError("times must be nondecreasing")
        width = len(self.states)
        # Each jump is credited to the first requested time at or after
        # it, so the running total is the occupancy at each of them.
        offset = np.searchsorted(time, self.times) * width
        size = (len(time) + 1) * width
        change = np.bincount(offset + self.targets, minlength=size)
        change -= np.bincount(offset + self.sources, minlength=size)
        cumulative = change.reshape(-1, width).cumsum(axis=0)
        counts = self.counts + cumulative[: len(time)]
        return np.asarray(counts, dtype=np.int64)

    def path(self, molecule: int) -> tuple[Floats, Ints]:
        """Times and states visited by one molecule, starting at zero."""
        jumps = self.molecules == molecule
        times = np.concatenate(([0.0], self.times[jumps]))
        visited = np.concatenate(
            (self.initial[molecule : molecule + 1], self.targets[jumps])
        )
        return times, visited

    def dwells(
        self, state: Selection, until: Selection | None = None
    ) -> Floats:
        """How long molecules stay in a state, pooled over every visit.

        Without a target the time is per visit, from arriving in the
        state to the next jump out of it, and is exponential with the
        total rate out. With one it is the time from arriving to first
        reaching the target, so an escape that comes straight back does
        not end the dwell; that time is the mean first passage time, and
        it is the longer of the two whenever escapes are recaptured.

        Dwells still running when the trajectory ends are dropped, which
        pulls the sample mean a little low.

        Args:
            state: State whose dwells are measured, or several treated
                as one.
            until: State that ends a dwell, or several. Defaults to the
                next jump out of state.

        Returns:
            One dwell time per completed visit, unordered.
        """
        inside = _indices(self.states, state)
        leaving = None if until is None else _indices(self.states, until)
        if leaving is not None and np.intersect1d(inside, leaving).size:
            raise ValueError("state and until must not share a state")

        order = np.argsort(self.molecules, kind="stable")
        times, targets = self.times[order], self.targets[order]
        edges = np.searchsorted(
            self.molecules[order], np.arange(len(self.initial) + 1)
        )
        spans: list[Floats] = []
        for molecule in range(len(self.initial)):
            low, high = edges[molecule], edges[molecule + 1]
            when = np.concatenate(([0.0], times[low:high]))
            visited = np.concatenate(
                (self.initial[molecule : molecule + 1], targets[low:high])
            )
            entered = np.flatnonzero(np.isin(visited, inside))
            if not entered.size:
                continue
            if leaving is None:
                # A visit ends at the next jump, which the last state
                # occupied does not have.
                closed = entered[entered + 1 < len(when)]
                spans.append(when[closed + 1] - when[closed])
            else:
                arrived = np.flatnonzero(np.isin(visited, leaving))
                spans.append(_first_passages(when, entered, arrived))
        if not spans:
            return np.zeros(0)
        return np.concatenate(spans)


def _indices(states: tuple[str, ...], selection: Selection) -> Ints:
    """State indices from names, indices, or a single name."""
    if isinstance(selection, str):
        selection = [selection]
    found = []
    for state in selection:
        if not isinstance(state, str):
            found.append(int(state))
        elif state in states:
            found.append(states.index(state))
        else:
            raise KeyError(f"unknown state {state!r}")
    return np.array(found, dtype=np.int64)


def _first_passages(when: Floats, entered: Ints, arrived: Ints) -> Floats:
    """Entry-to-arrival intervals, skipping entries already inside one."""
    spans: list[float] = []
    closed = 0
    for start in entered:
        if start < closed:
            continue
        index = int(np.searchsorted(arrived, start, side="right"))
        if index >= len(arrived):
            break  # still inside when the trajectory ended
        stop = int(arrived[index])
        spans.append(float(when[stop] - when[start]))
        closed = stop
    return np.array(spans, dtype=float)


def simulate(
    graph: Graph,
    initial: Amounts,
    duration: float,
    *,
    seed: int | np.random.Generator | None = None,
) -> Trajectory:
    """Simulate individual molecules moving between states.

    Args:
        graph: Graph to simulate.
        initial: Number of molecules starting in each state.
        duration: Simulate every molecule until it passes this time.
        seed: Seed or generator for the random number stream.

    Returns:
        The jump events of every molecule.
    """
    counts = np.asarray(graph.vector(initial))
    if (counts < 0).any() or not np.array_equal(np.rint(counts), counts):
        raise ValueError("initial counts must be nonnegative whole numbers")
    rng = np.random.default_rng(seed)
    log_rates = np.asarray(graph.log_rates)
    present = np.isfinite(log_rates)
    # Placeholder for absent transitions, which are masked out below
    log_rates = np.where(present, log_rates, 0.0)

    start = np.repeat(np.arange(len(graph)), counts.astype(np.int64))
    state = start.copy()
    time = np.zeros(len(state))
    events: list[tuple[Floats, Ints, Ints, Ints]] = []
    while True:
        live = np.flatnonzero(time < duration)
        if not live.size:
            break
        draws = rng.standard_exponential((live.size, len(graph)))
        waits = np.log(draws) - log_rates[state[live]]
        waits[~present[state[live]]] = np.inf
        target = waits.argmin(axis=1)
        wait = np.exp(waits[np.arange(live.size), target])
        # An infinite wait means the molecule is in an absorbing state
        moved = np.isfinite(wait)
        jumped = live[moved]
        time[live] += wait
        events.append((time[jumped], jumped, state[jumped], target[moved]))
        state[jumped] = target[moved]

    if not events:
        empty = np.zeros(0, dtype=np.int64)
        return Trajectory(graph.states, start, np.zeros(0), *(empty,) * 3)
    times, molecules, sources, targets = (
        np.concatenate(column) for column in zip(*events)
    )
    order = np.argsort(times, kind="stable")
    return Trajectory(
        states=graph.states,
        initial=start,
        times=times[order],
        molecules=molecules[order],
        sources=sources[order],
        targets=targets[order],
    )
