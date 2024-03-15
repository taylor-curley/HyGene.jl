# Hypothesis Generation (HyGene)

[![Build Status](https://github.com/taylor-curley/HypothesisGeneration.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/taylor-curley/HypothesisGeneration.jl/actions/workflows/CI.yml?query=branch%3Amaster)

A Julia implementation of a model of diagnostic hypothesis generation ("HyGene") and decision-making[^thomas2008][^thomas2014]. The overall model is based off of Minerva 2[^hintzman1986] and Minerva-DM[^dougherty1999].

## To-do

### Information

- [x] ~~Define high-level containers, like memory stores and information.~~
- [x] ~~Define `Context` with given number of features.~~
- [x] ~~Define generic method to build `Context` given certain arguments.~~
- [x] ~~Define `Trace` in LTM/SM with a given number of contexts.~~
- [x] ~~Define generic method to build `Trace` given certain arguments.~~
- [x] ~~Build `trace_replication` function for general vectors.~~
- [x] ~~Build `trace_replication` function for Context.~~
- [x] ~~Build `trace_replication` function for `Trace`.~~
- [x] ~~Build `trace_similarity` function for general vectors.~~
- [x] ~~Build `trace_similarity` function for `Trace` objects.~~
- [x] ~~Build `trace_decay!` function for `Context`.~~
- [x] ~~Build `trace_decay!` function for `Trace`.~~
- [ ] Write `Information` documentation.

### Long-term Memory

- [x] ~~Define `LongTermMemory` with a given number of episodic traces.~~
- [x] ~~Define generic method to build `LongTermMemory`~~
- [x] ~~Build `trace_decay!` function to pass through all items in LTM.~~
- [ ] Write `LongTermMemory` documentations.

### Semantic Memory

- [x] ~~Define `SemanticMemory` store.~~
- [x] ~~Define method for extracting unique events from LTM traces to build exemplars.~~
- [x] ~~Define method for replicating LTM and getting averages of feature vectors for exemplars.~~
- [ ] ~~Separate information objects for `SemanticMemory` traces?~~
- [ ] Write `SemanticMemory` documentations.

### Working Memory

- [ ] ~~Define `WorkingMemory` store.~~
- [ ] ~~Define functions necessary for `WorkingMemory`.~~
- [x] ~~Define `SetofContenders`.~~
- [x] ~~Define functions for `SetofContenders`.~~
- [ ] Write `SetofContenders` documentations.

### Helper functions

- [ ] Write documentation for all helper functions.

### Simulation Handler

- [ ] Define overall model object to house simulations and output.

# Footnotes and References

[^thomas2008]:
    Thomas, R. P., Dougherty, M. R., Sprenger, A. M., & Harbison, J. (2008). Diagnostic hypothesis generation and human judgment. _Psychological Review, 115_(1), 155-185. https://psycnet.apa.org/doi/10.1037/0033-295X.115.1.155
[^thomas2014]:
    Thomas, R. P., Dougherty, M. R., & Buttaccio, D. R. (2014). Memory constraints on hypothesis generation and decision making. _Current Directions in Psychological Science, 23_(4), 264-270. https://doi.org/10.1177/0963721414534853
[^hintzman1986]:
    Hintzman, D. L. (1986). "Schema abstraction" in a multiple-trace memory model. _Psychological Review, 93_(4), 411-428. https://psycnet.apa.org/doi/10.1037/0033-295X.93.4.411
[^dougherty1999]:
    Dougherty, M. R., Gettys, C. F., & Ogden, E. E. (1999). MINERVA-DM: A memory processes model for judgments of likelihood. _Psychological Review, 106_(1), 180. https://psycnet.apa.org/doi/10.1037/0033-295X.106.1.180