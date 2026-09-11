"""Time evolution of the master equation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from numpy import typing as npt

from .graph import Amounts, Graph
from .thermo import is_detailed_balanced, log_equilibrium


def propagate(
    graph: Graph,
    initial: Amounts,
    times: npt.ArrayLike,
    *,
    reversible: bool | None = None,
) -> Array:
    """Amount in each state over time, given an initial amount.

    Solves dp/dt = Q.T @ p in closed form. A reversible graph is solved
    by the symmetric eigendecomposition of Q, which stays exact however
    stiff the rate constants and however long the times. Any other graph
    is solved by a matrix exponential, which loses accuracy once the
    largest rate times the longest time exceeds about 1e9. Use
    ``integrate`` instead if the rate constants depend on time.

    Args:
        graph: Graph to solve.
        initial: Amount in each state at time zero, in any units; the
            total is conserved.
        times: Nonnegative times at which to report the amounts.
        reversible: Whether the graph is detailed balanced, as the
            eigendecomposition requires. Detected from the rate
            constants when not given, which needs concrete values; pass
            it to use propagate under jit, grad or vmap.

    Returns:
        Amounts, of shape (len(times), len(graph)).
    """
    start = graph.vector(initial)
    time = jnp.atleast_1d(jnp.asarray(times, dtype=float))
    if reversible is None:
        reversible = is_detailed_balanced(graph)
    if not reversible:
        generator = graph.generator().T
        exponential = jax.scipy.linalg.expm(
            generator * time[:, None, None], max_squarings=32
        )
        return jnp.matmul(exponential, start)

    # Centering the log equilibrium keeps sqrt(pi) and its inverse finite
    log_p = log_equilibrium(graph)
    log_p = log_p - (log_p.max() + log_p.min()) / 2
    values, vectors = jnp.linalg.eigh(symmetrized(graph))
    weights = vectors.T @ (start * jnp.exp(-log_p / 2))
    decay = jnp.exp(jnp.outer(time, jnp.minimum(values, 0)))
    modes = jnp.matmul(decay * weights, vectors.T)
    return modes * jnp.exp(log_p / 2)


def relaxation_rates(graph: Graph) -> Array:
    """Eigenvalues of -Q in ascending order, in inverse time.

    The first is zero and belongs to the equilibrium distribution; the
    rest are the rates of the exponential phases of any relaxation.
    Detailed balance is required, which makes the spectrum real.
    """
    if not is_detailed_balanced(graph):
        raise ValueError("relaxation rates require detailed balance")
    return jnp.sort(jnp.maximum(-jnp.linalg.eigvalsh(symmetrized(graph)), 0))


def relaxation_times(graph: Graph) -> Array:
    """Relaxation time of each nonequilibrium mode, slowest first."""
    return 1 / relaxation_rates(graph)[1:]


def symmetrized(graph: Graph) -> Array:
    """Q made symmetric by conjugation with sqrt(diag(pi)).

    Detailed balance gives sqrt(pi_i / pi_j) k_ij = sqrt(k_ij k_ji), so
    the transform is independent of the equilibrium distribution. The
    result has the same spectrum as Q and is safe to diagonalize.
    """
    log_rates = graph.log_rates
    coupling = jnp.exp((log_rates + log_rates.T) / 2)
    return coupling - jnp.diag(jnp.exp(log_rates).sum(axis=1))


def integrate(
    graph: Callable[[Any], Graph],
    initial: Amounts,
    times: npt.ArrayLike,
    *,
    solver: Any = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 2**16,
) -> Array:
    """Amount in each state over time, with time-dependent rates.

    Integrates dp/dt = Q(t).T @ p with Diffrax, for systems where the
    closed form of ``propagate`` does not apply: a ligand jump, a ramp,
    or any other driven rate constant. The default solver is implicit,
    since master equations are stiff whenever their timescales are well
    separated.

    Args:
        graph: The graph as a function of time, most easily written as
            ``lambda t: base.with_log_rates(...)``.
        initial: Amount in each state at the first time, in any units;
            the total is conserved.
        times: Increasing times at which to report the amounts; the
            first is the initial time.
        solver: Diffrax solver; defaults to ``diffrax.Kvaerno5()``.
        rtol: Relative tolerance of the step size controller.
        atol: Absolute tolerance of the step size controller.
        max_steps: Maximum number of steps taken.

    Returns:
        Amounts, of shape (len(times), len(graph)).
    """
    import diffrax  # optional dependency

    time = jnp.atleast_1d(jnp.asarray(times, dtype=float))
    start = graph(time[0]).vector(initial)
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, p, args: graph(t).generator().T @ p),
        solver if solver is not None else diffrax.Kvaerno5(),
        t0=time[0],
        t1=time[-1],
        dt0=None,
        y0=start,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(ts=time),
        max_steps=max_steps,
    )
    return jnp.asarray(solution.ys)
