# Hypothesis Generation (HyGene) Model

A Julia implementation of a model of diagnostic hypothesis generation ("HyGene") and decision-making[^thomas2008][^thomas2014]. The overall model is based off of Minerva 2[^hintzman1986] and Minerva-DM[^dougherty1999].

## Types and Structures

`HyGene.jl` simulations are centered upon the structures that hold the data and parameters. The values of each structure (`struct`) can be changes (`mutable`), and each structure belongs to a custom type hierarchy. All types and structures are defined within `./src/structures.jl`.

### `HypothesisGeneration`

The `HypothesisGeneration` object does not hold information; rather, it is a "supertype" that encompasses all other `struct`s in the simulation. For example, if you were to list the subtypes of `HypothesisGeneration`, it would look something like this:
```
julia> using TypeTree
julia> print(join(tt(HypothesisGeneration),""))
HypothesisGeneration
 ├─ Context
 ├─ HyGeneModel
 ├─ Hypothesis
 ├─ Observation
 └─ Trace
```
Thus, the rest of the structures listed below belong to the `HypothesisGeneration` superclass.

`HypothesisGeneration` is itself a subclass of a `ContinuousUnivariateDistribution`; however, this is mostly useful for use with Bayesian parameter estimators, such as `Turing.jl` and `DifferentialEvolutionMCMC.jl`.

### `HyGeneModel`

This is the main structure within the `HyGene.jl` architecture. It holds all static and dynamic parameters (although `mutable struct`s allow for redefinition), as well as memory stores (long-term, short-term, semantic). Information is written and directly manipulated from this structure. The information vectors that used in the simulations (such as `HyGeneModel.long_term_memory`) are compound vectors, or vectors of vectors. Each minivector represents a context or hypothesis, and several minivectors are chained together to form a larger representation.

### `Context` and `Hypothesis`

These are similar structures that hold singular vectors of numbers representing individual contexts and hypotheses. They are mostly useful when constructing observations and traces, but are also useful for comparison to degraded representations within a `HyGeneModel` object.

### `Observation` and `Trace`

A raw, unaltered version of an observed event is recorded as an `Observation`. These contain several mini-vectors, including one that provides information about the correct classification of an item (i.e., `Hypothesis`). When constructing a `HyGeneModel` object, `Observation` objects are translated into `Trace` objects, which contain degraded representations of the original item vectors with probability between 0 and 1 via `HyGeneModel.decay`. `Trace` objects are only meant to exist within memory stores in the main structure. When the model is engaged in a test trial, the contents of a new `Observation` object is compared to long-term memory.


## Simulation Controllers

There are a number of smaller functions that subserve the overall `HyGene.jl` architecture. They are listed here roughly by their function.

### Object Creation

  - `generate_item`: Returns a vector or collection of minivectors consisting of the integers `-1`, `0`, and `1` sampled with equal probability.
  - `generate_observation`: Returns an `Observation` object with an observed content vector or collection of minivectors.

### Object Replication

  - `trace_replicator`: Takes vector input and returns a vector with each number replaced with `-1`,`0`, or `1` with probability `similarity` [0.0,1.0].
  - `trace_decay`: Takes vector input and returns a vector with each number replaced with `0` with probability `decay` [0.0,1.0]. 
  - `trace_decay!`: Takes a simulation object and degrades the representation vector with probability `decay` [0.0,1.0].
  - `obs_to_trace`: Takes a "raw" `Observation` object, degrades the content vector with probability `decay`, and returns it within a `Trace` object.

### Object Comparison

  - `trace_similarity`: Calculates the dot product divided by the number of non-zero elements between two content vectors.
  - `trace_activation`: Calculates `trace_similarity` between two content vectors and raises the result to a given `power`.
  - `echo_intensity`: Returns the sum of all `trace_activation`s for a given `probe` against all items in a given memory store.
  - `echo_content`: Calculates `trace_activation` for all items against a `probe`, multiplies the items by their respective activation values, and returns the column sum of the vector or collection of minivectors.


## Example

A general model of the example from the Appendix of the original paper [^1] is available in `./model/example.jl`. 

# Footnotes and References

[^thomas2008]:
    Thomas, R. P., Dougherty, M. R., Sprenger, A. M., & Harbison, J. (2008). Diagnostic hypothesis generation and human judgment. _Psychological Review, 115_(1), 155-185. https://psycnet.apa.org/doi/10.1037/0033-295X.115.1.155
[^thomas2014]:
    Thomas, R. P., Dougherty, M. R., & Buttaccio, D. R. (2014). Memory constraints on hypothesis generation and decision making. _Current Directions in Psychological Science, 23_(4), 264-270. https://doi.org/10.1177/0963721414534853
[^hintzman1986]:
    Hintzman, D. L. (1986). "Schema abstraction" in a multiple-trace memory model. _Psychological Review, 93_(4), 411-428. https://psycnet.apa.org/doi/10.1037/0033-295X.93.4.411
[^dougherty1999]:
    Dougherty, M. R., Gettys, C. F., & Ogden, E. E. (1999). MINERVA-DM: A memory processes model for judgments of likelihood. _Psychological Review, 106_(1), 180. https://psycnet.apa.org/doi/10.1037/0033-295X.106.1.180