# Hypothesis Generation (HyGene)

A Julia implementation of a model of diagnostic hypothesis generation ("HyGene") and decision-making[^thomas2008][^thomas2014]. The overall model is based off of Minerva 2[^hintzman1986] and Minerva-DM[^dougherty1999].

## Types and Structures

`HyGene.jl` simulations are centered upon the structures that hold the data and parameters. The values of each structure (`struct`) can be changes (`mutable`), and each structure belongs to a custom type hierarchy. All types and structures are defined within `./src/structures.jl`.

### `HyGeneModel`

This is the main structure within the `HyGene.jl` architecture. It holds all static and dynamic parameters (although `mutable struct`s allow for redefinition), as well as memory stores (long-term, short-term, semantic). Information is written and directly manipulated from this structure. The information vectors that used in the simulations (such as `HyGeneModel.long_term_memory`) are compound vectors, or vectors of vectors. Each minivector represents a context or hypothesis, and several minivectors are chained together to form a larger representation.

# Footnotes and References

[^thomas2008]:
    Thomas, R. P., Dougherty, M. R., Sprenger, A. M., & Harbison, J. (2008). Diagnostic hypothesis generation and human judgment. _Psychological Review, 115_(1), 155-185. https://psycnet.apa.org/doi/10.1037/0033-295X.115.1.155
[^thomas2014]:
    Thomas, R. P., Dougherty, M. R., & Buttaccio, D. R. (2014). Memory constraints on hypothesis generation and decision making. _Current Directions in Psychological Science, 23_(4), 264-270. https://doi.org/10.1177/0963721414534853
[^hintzman1986]:
    Hintzman, D. L. (1986). "Schema abstraction" in a multiple-trace memory model. _Psychological Review, 93_(4), 411-428. https://psycnet.apa.org/doi/10.1037/0033-295X.93.4.411
[^dougherty1999]:
    Dougherty, M. R., Gettys, C. F., & Ogden, E. E. (1999). MINERVA-DM: A memory processes model for judgments of likelihood. _Psychological Review, 106_(1), 180. https://psycnet.apa.org/doi/10.1037/0033-295X.106.1.180