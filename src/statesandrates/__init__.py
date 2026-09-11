"""Graphs of states connected by rates, and analyses of them."""

import jax

# Rate constants and free energies span many orders of magnitude, which
# single precision cannot hold; JAX defaults to it unless told otherwise
jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

__version__ = "0.2.0"

__all__ = [
    "Graph",
    "Mutation",
    "Scheme",
    "Trajectory",
    "absorption_probabilities",
    "barrier_energies",
    "coarse_grain",
    "cycle_affinity",
    "dwell_diagram",
    "dwell_rates",
    "energy_diagram",
    "equilibrium",
    "fluxes",
    "free_energies",
    "integrate",
    "is_detailed_balanced",
    "is_lumpable",
    "log_equilibrium",
    "mean_first_passage_times",
    "mean_residence_times",
    "passage_rates",
    "propagate",
    "relaxation_rates",
    "relaxation_times",
    "simulate",
    "state_diagram",
    "symmetrized",
    "thermal_energy",
]
from .graph import Graph, Mutation, thermal_energy
from .kinetics import (
    integrate,
    propagate,
    relaxation_rates,
    relaxation_times,
    symmetrized,
)
from .macro import (
    absorption_probabilities,
    coarse_grain,
    dwell_rates,
    is_lumpable,
    mean_first_passage_times,
    mean_residence_times,
    passage_rates,
)
from .plotting import dwell_diagram, energy_diagram, state_diagram
from .scheme import Scheme
from .simulate import Trajectory, simulate
from .thermo import (
    barrier_energies,
    cycle_affinity,
    equilibrium,
    fluxes,
    free_energies,
    is_detailed_balanced,
    log_equilibrium,
)
