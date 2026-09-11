"""Diagrams of a graph and of its free energy landscape."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy import typing as npt

from .graph import Edge, Graph
from .macro import dwell_rates
from .scheme import Scheme
from .thermo import barrier_energies, free_energies

Floats = npt.NDArray[np.floating[Any]]


def energy_diagram(
    graph: Graph,
    path: Sequence[str] | None = None,
    *,
    reference: str | None = None,
    prefactor: float = 1.0,
    width: float = 0.5,
    resolution: int = 16,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Line2D:
    """Draw the free energy landscape along a path of states.

    States are drawn as level segments at their free energy, joined over
    the barrier of the transition between them. The diagram is a single
    line, so graphs can be overlaid on one axis by drawing them in turn.

    Args:
        graph: Graph to draw; must be detailed balanced.
        path: States to draw in order, each consecutive pair of which
            must interconvert. Defaults to every state in graph order.
        reference: State taken as the zero of energy.
        prefactor: Rate constant of a barrierless transition, which
            shifts every barrier by RT log of itself.
        width: Fraction of the state spacing taken up by each state.
        resolution: Number of points drawn over each half of a barrier.
        ax: Axis to draw on; defaults to the current axis.
        **kwargs: Passed to ``Axes.plot``.

    Returns:
        The line of the diagram.
    """
    states = tuple(graph.states if path is None else path)
    energies = np.asarray(free_energies(graph, reference=reference))
    barriers = np.asarray(
        barrier_energies(graph, reference=reference, prefactor=prefactor)
    )
    steps = [graph.index(state) for state in states]

    x: list[Floats] = [np.array([-width / 2, width / 2])]
    y: list[Floats] = [np.full(2, energies[steps[0]])]
    for step, (i, j) in enumerate(zip(steps, steps[1:])):
        if not np.isfinite(barriers[i, j]):
            raise ValueError(
                f"no transition between {states[step]!r}"
                f" and {states[step + 1]!r}"
            )
        fraction = np.linspace(0, 1, 2 * resolution + 1)
        x.append(step + width / 2 + fraction * (1 - width))
        y.append(_hump(energies[i], barriers[i, j], energies[j], fraction))
        x.append(step + 1 + np.array([-width / 2, width / 2]))
        y.append(np.full(2, energies[j]))

    ax = plt.gca() if ax is None else ax
    (line,) = ax.plot(np.concatenate(x), np.concatenate(y), **kwargs)
    ax.set_xticks(range(len(states)), states)
    ax.set_ylabel("free energy (kcal/mol)")
    return line


def state_diagram(
    graph: Graph | Scheme,
    pos: Mapping[str, npt.ArrayLike] | None = None,
    *,
    rate_labels: bool = True,
    fmt: str = "{:.3g}",
    curvature: float = 0.15,
    node_size: float | None = None,
    node_color: Any = "white",
    font_size: float = 10.0,
    ax: Axes | None = None,
    **kwargs: Any,
) -> dict[str, Floats]:
    """Draw the states of a graph as nodes and its transitions as arrows.

    Each transition is drawn as a curved arrow from its source to its
    target, so that both directions of a reversible pair are visible,
    and is labelled with its rate constant. A scheme has no values, so
    its arrows are labelled with the names of its rate constants.

    Args:
        graph: Graph or scheme to draw.
        pos: Position of each state; defaults to a spring layout, which
            is deterministic but arbitrary. Pass the returned positions
            back in to draw a related graph on the same layout.
        rate_labels: Whether to label each arrow with its rate constant.
        fmt: Format of a rate constant label; ignored for a scheme,
            whose labels are names rather than numbers.
        curvature: Bend of each arrow, as a fraction of its length; zero
            draws straight arrows, which overlap in a reversible pair. A
            positive value puts the left-to-right arrow of a pair above
            its partner; negative swaps them.
        node_size: Area of a node in points squared; defaults to
            whatever fits the longest state name.
        node_color: Color of the nodes, either one color or one for each
            state in graph order.
        font_size: Size of the state and rate constant labels.
        ax: Axis to draw on; defaults to the current axis.
        **kwargs: Passed to ``networkx.draw_networkx_edges``, such as
            ``width`` or ``edge_color``. A sequence of either is taken
            in the order ``graph.edges`` gives.

    Returns:
        The position of each state.
    """
    import networkx as nx  # optional dependency

    if node_size is None:
        longest = max((len(state) for state in graph.states), default=1)
        node_size = (0.62 * font_size * longest + 6.0) ** 2

    digraph = nx.DiGraph()
    digraph.add_nodes_from(graph.states)
    digraph.add_edges_from(graph.edges)

    layout: dict[str, Floats]
    if pos is None:
        layout = {
            state: np.asarray(point, dtype=float)
            for state, point in nx.spring_layout(digraph, seed=0).items()
        }
    else:
        if missing := set(graph.states) - set(pos):
            raise ValueError(f"no position for {sorted(missing)}")
        layout = {
            state: np.asarray(pos[state], dtype=float)
            for state in graph.states
        }

    ax = plt.gca() if ax is None else ax
    # Negated so that a positive curvature puts the left-to-right arrow of
    # a reversible pair on top, which is how such pairs are usually drawn
    connectionstyle = f"arc3,rad={-curvature}"
    nx.draw_networkx_nodes(
        digraph,
        layout,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        edgecolors="black",
    )
    nx.draw_networkx_labels(digraph, layout, ax=ax, font_size=font_size)
    nx.draw_networkx_edges(
        digraph,
        layout,
        # Given explicitly so that a per-transition width or color is
        # taken in the order graph.edges reports, rather than whatever
        # order the underlying graph happens to store
        edgelist=list(graph.edges),
        ax=ax,
        node_size=node_size,
        connectionstyle=connectionstyle,
        **kwargs,
    )
    if rate_labels:
        nx.draw_networkx_edge_labels(
            digraph,
            layout,
            _edge_labels(graph, fmt),
            ax=ax,
            font_size=font_size,
            connectionstyle=connectionstyle,
        )
    points = np.stack([layout[state] for state in graph.states])
    _fit_nodes(ax, points, node_size, _sag(layout, graph.edges, curvature))
    ax.set_axis_off()
    return layout


def dwell_diagram(
    graph: Graph,
    dwells: npt.ArrayLike,
    state: str | int,
    until: str | Sequence[str] | None = None,
    *,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Draw sampled dwell times against the rates they should report.

    The histogram is the sample and each line is the density of an
    exponential at one of the rates dwell_rates gives, so a dwell
    measured to a target should follow the observable rate and one
    measured per visit should follow the microscopic rate. Which line
    the sample follows is the point of the figure.

    Args:
        graph: Graph the dwells were simulated from.
        dwells: Sampled dwell times, from Trajectory.dwells.
        state: State the dwells are in.
        until: State that ended each dwell, if any. Left out, only the
            microscopic rate is drawn.
        ax: Axis to draw on. Defaults to the current one.
        **kwargs: Passed to ``Axes.hist``.

    Returns:
        The axis drawn on.
    """
    axis = plt.gca() if ax is None else ax
    axis.hist(
        np.asarray(dwells, dtype=float),
        **{"bins": 30, "color": "0.85", "density": True, **kwargs},
    )
    grid = np.linspace(0.0, axis.get_xlim()[1], 200)
    for name, rate in dwell_rates(graph, state, until).items():
        axis.plot(
            grid,
            rate * np.exp(-rate * grid),
            "-" if name == "observable" else "--",
            lw=1,
            label=f"{name}, {rate:.3g}",
        )
    axis.set(xlabel="dwell time", ylabel="density")
    return axis


def _edge_labels(graph: Graph | Scheme, fmt: str) -> dict[Edge, str]:
    """Text to draw on each arrow: a rate constant, or its name."""
    if isinstance(graph, Scheme):
        return {
            edge: graph.label(name) for edge, name in graph.symbols.items()
        }
    rates = np.asarray(graph.rates)
    return {
        (source, target): fmt.format(
            rates[graph.index(source), graph.index(target)]
        )
        for source, target in graph.edges
    }


def _sag(
    layout: dict[str, Floats], edges: Sequence[Edge], curvature: float
) -> float:
    """How far the most bowed arrow strays from the line it spans.

    An arc3 arrow reaches half of its control point offset, which is the
    curvature times its length, so the deepest bow is that of the
    longest transition.
    """
    lengths = [
        float(np.hypot(*(layout[source] - layout[target])))
        for source, target in edges
    ]
    return abs(curvature) / 2 * max(lengths, default=0.0)


def _fit_nodes(
    ax: Axes, points: Floats, node_size: float, pad: float = 0.0
) -> None:
    """Widen the limits of an axis until whole nodes fit inside it.

    Nodes are drawn at a size in points, but the limits only span their
    centers, so those at the edge of the layout are clipped in half. The
    room a node needs depends on the limits it is asking to widen, hence
    the fraction below rather than a plain sum. The pad is further room
    in data units, for whatever is drawn outside the nodes themselves.
    """
    figure = ax.get_figure()
    if figure is None:  # an axis detached from its figure has no size
        return
    corners = ax.transAxes.transform([(0.0, 0.0), (1.0, 1.0)])
    size = (corners[1] - corners[0]) * 72 / figure.dpi
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    fraction = np.clip((np.sqrt(node_size) + 4.0) / size, 0.0, 0.8)
    half = span / (1 - fraction) / 2 + pad
    center = (low + high) / 2
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])


def _hump(start: float, peak: float, end: float, fraction: Floats) -> Floats:
    """Rise from start to peak and fall to end, flat at each end."""
    rise = _smoothstep(np.clip(2 * fraction, 0, 1))
    fall = _smoothstep(np.clip(2 * fraction - 1, 0, 1))
    return start + (peak - start) * rise + (end - peak) * fall


def _smoothstep(fraction: Floats) -> Floats:
    """Smooth interpolation from 0 to 1 with zero slope at both ends."""
    return fraction * fraction * (3 - 2 * fraction)
