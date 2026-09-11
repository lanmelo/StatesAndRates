# StatesAndRates
Graphs of states connected by rates, and analyses of them

A model is a set of named states and the first-order rate constants
between them. Mutations perturb those rates through the free energies of
the states and of the barriers between them, so a mutant graph stays
thermodynamically consistent. Graphs support:

- equilibrium populations and the free energy landscape behind them,
- stationary fluxes and cycle affinities, for graphs that are driven,
- passage times, residence times and the probability of each fate,
- the exact time evolution of any starting population,
- macroscopic rates, for when some states cannot be resolved,
- Gillespie simulation of individual molecules,
- diagrams of the graph itself and of its energy landscape.

# Installation and Usage
Install with `pip install -e .`, adding `.[driven]` for time-dependent
rate constants, which need Diffrax, and `.[graph]` for graph diagrams,
which need NetworkX.

```python
import numpy as np
import statesandrates as sr

# A transcription factor that must find its site before binding it
graph = sr.Graph({
    ("free", "search"): 1e2, ("search", "free"): 5e1,
    ("search", "bound"): 2e2, ("bound", "search"): 1e-1,
})

sr.equilibrium(graph)               # populations, by state reduction
sr.free_energies(graph, reference="free")   # kcal/mol
sr.relaxation_times(graph)          # the observable kinetic phases
sr.propagate(graph, {"free": 1.0}, np.linspace(0, 1, 100))

# The searching intermediate is invisible, so lump it with the free state
groups = {"apo": ["free", "search"], "bound": ["bound"]}
sr.passage_rates(graph, groups).rates   # the k_on and k_off measured

# Where a passage time is actually spent, state by state, and how often
# each state is revisited before the target is reached
sr.mean_first_passage_times(graph, "free")     # the totals
sr.mean_residence_times(graph, "free")         # the same, split by state

# Which of two fates comes first, which a passage time cannot say
sr.absorption_probabilities(graph, ["free", "bound"])

# A chain is detailed balanced whatever its rates, but a cycle need not
# be. The flux says how fast it circulates, the affinity says what drives
# it, and both vanish exactly at equilibrium.
cycle = sr.Graph({
    ("a", "b"): 2.0, ("b", "a"): 1.0, ("b", "c"): 3.0,
    ("c", "b"): 4.0, ("c", "a"): 5.0, ("a", "c"): 1.5,
})
sr.fluxes(cycle)                            # net flux through each transition
sr.cycle_affinity(cycle, ["a", "b", "c"])   # kcal/mol dissipated per turn

# A mutation that raises the barrier to binding by 1.5 kcal/mol
mutant = graph.mutate(sr.Mutation(barriers={("search", "bound"): 1.5}))
sr.energy_diagram(graph, label="wild type")
sr.energy_diagram(mutant, label="mutant")

# The whole model at a glance: states as nodes, rates on the arrows
sr.state_diagram(graph)

# A mechanism can be written once with its rates named rather than valued.
# A scheme holds no numbers, so no analysis will accept it by mistake; it
# instantiates into an ordinary Graph, and its diagram shows the names.
scheme = sr.Scheme(
    {("free", "search"): "k_on", ("search", "free"): "k_off",
     ("search", "bound"): "k_rec", ("bound", "search"): "k_esc"},
    latex={"k_on": r"$k_\mathrm{on}$"},
)
sr.state_diagram(scheme)            # arrows labelled k_on, k_off, ...
wild = scheme.graph({"k_on": 1e6, "k_off": 5.0, "k_rec": 3.0, "k_esc": 0.5})
scheme.graph(wild, k_esc=5.0)       # a variant, naming only what differs

# Individual molecules
trajectory = sr.simulate(graph, {"free": 1000}, duration=1.0, seed=0)
trajectory.occupancy(np.linspace(0, 1, 100))
trajectory.path(0)

# How long they stay, and the two rates a dwell can report: the time per
# visit follows the total rate out, while the time until the molecule
# actually leaves follows the mean first passage rate, which is slower
# whenever an escape is usually recaptured
trajectory.dwells("bound")                      # per visit
trajectory.dwells("bound", until="free")         # until it really goes
sr.dwell_rates(graph, "bound", "free")           # both expectations
sr.dwell_diagram(graph, trajectory.dwells("bound", until="free"),
                 "bound", "free")                # sample against both
```

Every function's documentation is [in the code](src/statesandrates), or
available through Python's `help()`.

# Documentation
[docs/marklund2022.ipynb](docs/marklund2022.ipynb) works every feature
through the *lac* repressor model of Marklund *et al.*, Science 375,
442 (2022), using the rates and energies that paper reports: the free
energy landscape of three lac operators, the macroscopic rates an
experiment can resolve, mutant libraries, flow-cell kinetics,
single-molecule traces, and fitting microscopic rates back to
macroscopic ones. Install its extras with `pip install -e .[docs]`.

[docs/presentation.pdf](docs/presentation.pdf) is a slide version. It opens
with the argument the package is built around, that a kinetic scheme is a
falsifiable hypothesis rather than a summary of data, and then covers the
theory and the numerics: detailed balance, the GTH reduction for
equilibrium, energy diagrams, the symmetrized generator, mean first passage
times as a linear solve, coarse graining, Gillespie sampling and dwell
times, illustrated with figures from that notebook's model. Rebuild it with

```bash
cd docs
jupyter nbconvert --to notebook --execute --inplace marklund2022.ipynb
latexmk -xelatex presentation.tex
```

`xelatex` rather than `pdflatex`, since the slides set Arial through
`fontspec`.

The notebook writes the figures the slides include, so the model is built
and plotted in exactly one place.

[docs/michaelis_menten.ipynb](docs/michaelis_menten.ipynb) derives the
Michaelis-Menten rate law from a three-state cycle rather than assuming
it, and works the rest of the package through the same graph: the
commitment to catalysis, the pre-steady-state burst and how a mixing
dead time distorts it, single-enzyme traces, mutants expressed in
kcal/mol, flux control coefficients, and the product concentration at
which the enzyme stalls and runs backwards.

[docs/proofreading.ipynb](docs/proofreading.ipynb) builds Hopfield's
kinetic proofreading scheme as he wrote it, four states and five
transitions, and gets the error as a product of two branch
probabilities: `1/f^2` rather than the `1/f` that binding energy alone
allows. It goes on to what that costs in turnover, how it scales with
the number of checkpoints, and what the one-way arrows are worth in
kcal/mol once the reverse steps are put back.

[docs/folding_and_binding.ipynb](docs/folding_and_binding.ipynb) runs a
double mutant cycle on a protein that has to fold before it can bind.
Two mutations that are additive in stability to the last bit, and that
never touch the binding interface, produce 1.26 kcal/mol of apparent
coupling in binding data; the notebook covers which fit parameter that
lands in, `K_d` or `R_max`, depending on the assay rather than on the
protein, the hundredfold spread it puts in the observed off-rate,
why the observed binding rate cannot exceed the folding rate, and the
geometry behind all of it: the variants are exactly additive in a plane of
stability against interface energy, and the coupling appears only because
the measured affinity is a curved function of that plane.

[docs/references.md](docs/references.md) lists the source for each method
the package implements, grouped by the function that uses it.

# Notes
Rate constants are held as logarithms and analyses avoid subtracting
quantities of similar size, so rates spanning many orders of magnitude
stay accurate. Equilibrium populations come from Grassmann-Taksar-Heyman
state reduction, and spectra and time courses from a symmetrized
generator rather than from a matrix exponential, which loses accuracy
for stiff rates over long times.

The array math is JAX, so a graph is a pytree that can be passed through
`jit`, `grad` and `vmap`: rate constants can be fit to data by gradient
descent, and mutant libraries can be evaluated in a single batch. Use
`Graph.with_log_rates` to swap in traced rates. Importing the package
enables double precision, which free energies need. The Gillespie
simulator is NumPy, since its trajectories are ragged.

Detailed balance is required only where it is physically needed. A
stationary distribution exists for any irreducible graph, so
`equilibrium`, `fluxes`, `propagate` and the passage times all work on a
driven one; `barrier_energies`, `energy_diagram` and `relaxation_times`
refuse, since a driven transition has no single barrier height and no
real spectrum.

Two coarse grainings are offered because a macroscopic rate is not
unique. `coarse_grain` averages the microscopic rates over the local
equilibrium, reproducing equilibrium populations and fluxes;
`passage_rates` inverts mean first passage times, reproducing measured
dwell times. They agree when the partition is lumpable, which
`is_lumpable` tests.
