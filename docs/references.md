# References
Where the methods in this package come from, and what to read to
understand each one. Grouped by the feature it underpins rather than by
topic, so a function can be traced to its source.

# Equilibrium populations, stationary flux and cycle affinities
`equilibrium`, `log_equilibrium`, `fluxes`, `cycle_affinity`

Nam K-M, Martinez-Corral R, Gunawardena J. The linear framework: using
graph theory to reveal the algebra and thermodynamics of biomolecular
systems. *Interface Focus* **12**, 20220013 (2022).
[doi:10.1098/rsfs.2022.0013](https://doi.org/10.1098/rsfs.2022.0013)

> Read this one first. It is the closest thing to a manual for the model
> this package implements: finite directed graphs of states with rate
> constants on the edges, aimed at people building kinetic models of
> biomolecules. The stationary distribution comes out as a spanning-tree
> formula through the matrix-tree theorem, which is the closed form
> behind `equilibrium`, and detailed balance and free energies are
> developed on the same graph. Open access.

Mirzaev I, Gunawardena J. Laplacian dynamics on general graphs.
*Bulletin of Mathematical Biology* **75**, 2118-2149 (2013).
[doi:10.1007/s11538-013-9884-8](https://doi.org/10.1007/s11538-013-9884-8)

> The proofs behind the review above, including graphs that are not
> strongly connected.

King EL, Altman C. A schematic method of deriving the rate laws for
enzyme-catalyzed reactions. *Journal of Physical Chemistry* **60**,
1373-1378 (1956).
[doi:10.1021/j150544a010](https://doi.org/10.1021/j150544a010)

> The original diagrammatic method for the steady state of a kinetic
> scheme, and the version biochemists still cite.

Hill TL. *Free Energy Transduction and Biochemical Cycle Kinetics*.
Springer (1989); Dover reprint (2005). Chapter 2.

> The diagram method together with the free energy levels of the states
> in a kinetic diagram, which is the object `free_energies` returns and
> `energy_diagram` draws. Also the clearest account of why cycles, not
> individual transitions, carry the thermodynamics.

Schnakenberg J. Network theory of microscopic and macroscopic behavior
of master equation systems. *Reviews of Modern Physics* **48**, 571-585
(1976).
[doi:10.1103/RevModPhys.48.571](https://doi.org/10.1103/RevModPhys.48.571)

> The same construction from the physics side, via Kirchhoff's theorem,
> extended to cycle affinities and driven steady states. This is the
> reference for `cycle_affinity` and for reading a nonzero `fluxes` as
> circulation around a cycle rather than as a rate.

# The GTH reduction
`thermo._gth`

Grassmann WK, Taksar MI, Heyman DP. Regenerative analysis and steady
state distributions for Markov chains. *Operations Research* **33**,
1107-1116 (1985).
[doi:10.1287/opre.33.5.1107](https://doi.org/10.1287/opre.33.5.1107)

> State reduction using only the nonnegative off-diagonal rates, so no
> subtractive cancellation can occur. This is what `equilibrium` runs.

Stewart WJ. *Introduction to the Numerical Solution of Markov Chains*.
Princeton University Press (1994).

> GTH placed among its alternatives, with the argument for preferring
> it, plus a chapter on transient solutions that covers `propagate` and
> `integrate`.

# Detailed balance
`is_detailed_balanced`, `barrier_energies`

Kelly FP. *Reversibility and Stochastic Networks*. Wiley (1979);
Cambridge University Press (2011). Chapter 1.
[Free from the author](https://www.statslab.cam.ac.uk/~fpk1/BOOKS/kelly_book.html)

> Kolmogorov's criterion, that log(k_ij / k_ji) sums to zero around
> every cycle, which is the condition `is_detailed_balanced` tests.

# First passage times, residence times and fates
`mean_first_passage_times`, `mean_residence_times`,
`absorption_probabilities`, `dwell_rates`, `passage_rates`,
`is_lumpable`

Bar-Haim A, Klafter J. On mean residence and first passage times in
finite one-dimensional systems. *Journal of Chemical Physics* **109**,
5187-5193 (1998).
[doi:10.1063/1.477135](https://doi.org/10.1063/1.477135)

> Mean residence times and mean first passage times derived together,
> and the distinction between them: a first passage time is a sum of
> residence times, so it exceeds the time per visit whenever an escape
> is recaptured. That is the difference `dwell_rates` reports as its
> microscopic and observable rates.

Kemeny JG, Snell JL. *Finite Markov Chains*. Springer (1976).
Chapter 4 and section 6.3.

> Chapter 4 gives the fundamental matrix, which is exactly what
> `mean_residence_times` computes, and the absorption probabilities that
> `absorption_probabilities` returns: the inverse of the generator
> restricted to the non-target states, whose row sums are the passage
> times and whose entries divided by the time per visit are visit
> counts. Section 6.3 gives the lumpability condition that
> `is_lumpable` checks.

Nam K-M, Gunawardena J. Algebraic formulas for first-passage times of
Markov processes in the linear framework. *Bulletin of Mathematical
Biology* **87**, 161 (2025).
[doi:10.1007/s11538-025-01524-z](https://doi.org/10.1007/s11538-025-01524-z)

> Passage times as spanning-forest formulas, in the same idiom as the
> *Interface Focus* review. Same mathematics as Bar-Haim and Klafter,
> algebraic rather than analytic.

# Gillespie simulation
`simulate`, `Trajectory`

Gillespie DT. A general method for numerically simulating the stochastic
time evolution of coupled chemical reactions. *Journal of Computational
Physics* **22**, 403-434 (1976).
[doi:10.1016/0021-9991(76)90041-3](https://doi.org/10.1016/0021-9991(76)90041-3)

> The original, and the paper that derives the first reaction method
> `simulate` uses: draw a waiting time for every transition out of the
> current state and take the shortest.

Gillespie DT. Exact stochastic simulation of coupled chemical reactions.
*Journal of Physical Chemistry* **81**, 2340-2361 (1977).
[doi:10.1021/j100540a008](https://doi.org/10.1021/j100540a008)

> The direct method, and the more cited of the two.

Gillespie DT. Stochastic simulation of chemical kinetics. *Annual Review
of Physical Chemistry* **58**, 35-55 (2007).
[doi:10.1146/annurev.physchem.58.032806.104637](https://doi.org/10.1146/annurev.physchem.58.032806.104637)

> The review, and the reference for the scope of the algorithm as
> against the scope of this implementation: it sets up propensities for
> bimolecular channels from the start, so the algorithm is general even
> though `simulate` is restricted to first-order transitions.

# Coarse graining and the symmetrized generator
`coarse_grain`, `symmetrized`, `relaxation_rates`, `relaxation_times`

Prinz J-H, Wu H, Sarich M, Keller B, Senne M, Held M, Chodera JD,
Schutte C, Noe F. Markov models of molecular kinetics: generation and
validation. *Journal of Chemical Physics* **134**, 174105 (2011).
[doi:10.1063/1.3565032](https://doi.org/10.1063/1.3565032)

> Estimation under a detailed balance constraint, the symmetrized
> eigenproblem that makes the spectrum real, and how to test whether a
> lumped model reproduces the kinetics it replaced.

# The worked example
Marklund E, Mao G, Yuan J, Zikrin S, Abdurakhmanov E, Deindl S, Elf J.
Sequence specificity in DNA binding is mainly governed by association.
*Science* **375**, 442-445 (2022).
[doi:10.1126/science.abg7427](https://doi.org/10.1126/science.abg7427)

> The source of the rates, energies and three-state searching / testing
> / bound mechanism used throughout
> [marklund2022.ipynb](marklund2022.ipynb) and
> [presentation.pdf](presentation.pdf).

Hopfield JJ. Kinetic proofreading: a new mechanism for reducing errors
in biosynthetic processes requiring high specificity. *PNAS* **71**,
4135-4139 (1974).
[doi:10.1073/pnas.71.10.4135](https://doi.org/10.1073/pnas.71.10.4135)

Ninio J. Kinetic amplification of enzyme discrimination. *Biochimie*
**57**, 587-595 (1975).
[doi:10.1016/S0300-9084(75)80139-8](https://doi.org/10.1016/S0300-9084(75)80139-8)

> The mechanism built in [proofreading.ipynb](proofreading.ipynb).
> Hopfield's paper is where the error falls to the square of the
> equilibrium value, and where the free energy that costs is identified
> as the price of the accuracy.

# Double mutant cycles and coupled equilibria
The subject of [folding_and_binding.ipynb](folding_and_binding.ipynb).

Carter PJ, Winter G, Wilkinson AJ, Fersht AR. The use of double mutants
to detect structural changes in the active site of the tyrosyl-tRNA
synthetase (*Bacillus stearothermophilus*). *Cell* **38**, 835-840
(1984).
[doi:10.1016/0092-8674(84)90278-2](https://doi.org/10.1016/0092-8674(84)90278-2)

> Where the double mutant cycle comes from, and the argument that a
> nonadditive residual reports an interaction between the two sites.

Horovitz A. Double-mutant cycles: a powerful tool for analyzing protein
structure and function. *Folding and Design* **1**, R121-R126 (1996).
[doi:10.1016/S1359-0278(96)00056-9](https://doi.org/10.1016/S1359-0278(96)00056-9)

Pagano L, Toto A, Malagrino F, Visconti L, Jemth P, Gianni S. Double
mutant cycles as a tool to address folding, binding, and allostery.
*International Journal of Molecular Sciences* **22**, 828 (2021).
[doi:10.3390/ijms22020828](https://doi.org/10.3390/ijms22020828)

> Two reviews of the method, the second specifically on applying it to
> systems where folding and binding are coupled, which is the case the
> notebook works through.

Wyman J, Gill SJ. *Binding and Linkage: Functional Chemistry of
Biological Macromolecules*. University Science Books (1990).

> The thermodynamics of linked equilibria, which is what makes a
> mutation that only touches stability show up in a binding
> measurement.

Otwinowski J, McCandlish DM, Plotkin JB. Inferring the shape of global
epistasis. *PNAS* **115**, E7550-E7558 (2018).
[doi:10.1073/pnas.1804015115](https://doi.org/10.1073/pnas.1804015115)

> The general form of the problem: a nonlinear map from an additive
> underlying quantity to the measured one generates apparent epistasis
> everywhere, and can be separated from the real thing given enough
> variants. The notebook's residual is one instance, with the nonlinear
> map known exactly rather than inferred.

# Shortest path through the above
*Interface Focus* (2022) for where the equilibrium distribution comes
from, Stewart (1994) for how it is actually computed, Hill chapter 2 for
what the energy diagram means, Bar-Haim and Klafter (1998) for passage
versus residence times, and Gillespie (2007) for simulation.
