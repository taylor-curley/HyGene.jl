"""
    HypothesisGeneration

An abstract type representing all structures related to HyGene simulations. 
Mostly useful when using custom samplers, such as with `Turing.jl`, but also
for preventing excessive function re-definition for different sub-types/structures.
All `HypothesisGeneration` structures have a field called `content` that hold
vectors/minivectors representing information.

# Subtypes

  - `HyGeneModel`
  - `Observation`
  - `Context`
  - `Hypothesis`
  - `Trace`

"""
abstract type HypothesisGeneration <: ContinuousUnivariateDistribution end


"""
    HyGeneModel

Mutable structure that holds all parameters and data related to HyGene simulations. 
This not only includes hyperparameters, but long-term, working, and semantic memory
stores. 

# Arguments

  - `n_values`: Length of feature mini-vectors.
  - `n_contexts`: Number of mini-vectors within a single item.
  - `n_lt_memory`: How many items in long-term memory?
  - `n_sem_memory`: How many items in semantic memory?
  - `act_thresh`: Minimum activation of a trace in LTM needed for derivation of unspecified probe.
  - `similarity`: Proportion of shared integers within similar items. Can be dynamic for different contexts.
  - `decay`: Proportion of features that turn "off" with trace decay.
  - `t_max`: Maximum number of failed retrieval attempts.
  - `ϕ`: Maximum number of items in working memory.
  - `act_min_reset`: Value to reset `act_min_h` to. Original paper specifies `0.0`, but this could be an interesting manipulation.
  - `trial`: How many trials have elapsed?.
  - `act_min_h`: Minimum activation an item must have to enter set of contenders (SOC).
  - `t`: Current number of retrieval attempts.
  - `contexts`: List of contextual information, i.e. `Context` objects. Not actually a part of memory.
  - `semantic_memory`: Semantic memory store. Basically a vector of `Hypothesis` objects. Questionable if semantic memory is static, but I will assume so.
  - `working_memory`: Working memory store. Initially empty, but will hold `Hypothesis` objects as part of the Set of Contenders (SOC).
  - `long_term_memory`: Long-term memory store. Basically a vector of `Trace` objects.

"""
Base.@kwdef mutable struct HyGeneModel <: HypothesisGeneration
    # Descriptive data
    n_values          = 15
    n_contexts        = 4
    n_lt_memory       = 40
    n_sem_memory      = 40
    # Static parameters
    act_thresh        = 0.1
    similarity        = 0.5
    decay             = 0.1
    t_max             = 10
    ϕ                 = 4
    act_min_reset     = 0.0
    # Variable parameters
    trial             = 0
    act_min_h         = 0.0
    t                 = 0
    # Static stores
    contexts          = Vector{Context}(undef,0)
    semantic_memory   = Vector{Hypothesis}(undef,0)
    # Variable stores
    working_memory    = Vector{HypothesisGeneration}(undef,0)
    long_term_memory  = Vector{Trace}(undef,0)
end


"""
    Observation

A single "true" (unaltered) observation. Will be written to episodic memory with
some level of trace decay. Observations cannot be constructed with null `content`.

# Arguments

  - `label`: Text label. (Mostly for house-keeping purposes.)
  - `n_values`: Length of feature mini-vectors.
  - `n_contexts`: Number of mini-vectors within a single item.
  - `content`: Vector (or mini-vectors if `n_features > 1`) of values representing the item.

"""
Base.@kwdef mutable struct Observation <: HypothesisGeneration
    # Static descriptors
    label
    n_values
    n_contexts
    # Variable entities
    content
end


"""
    Context

Holds information about a single context within the HyGene simulation. Contexts are 
chained together to comprise the content of a Trace.

# Arguments

  - `label`: Text label. (Mostly for house-keeping purposes.)
  - `n_values`: Length of feature mini-vectors.
  - `content`: Vector of values of length `n_values`.

# Constructor

  - `Context(label,n_values)`: If a `content` vector is not supplied, it will be created.

"""
Base.@kwdef mutable struct Context <: HypothesisGeneration
    label
    n_values
    content
    # Optional constructor for null content
    Context(label,n_values) = new(label,n_values,generate_item(n_values))
end


"""
    Hypothesis

Holds information about a single hypothesis within the HyGene simulation. Hypotheses
are kept in semantic memory and appended to Trace content. Hypotheses cannot be 
constructed with null `content`.

# Arguments

  - `label`: Text label. (Mostly for house-keeping purposes.)
  - `n_values`: Length of feature mini-vectors.
  - `content`: Vector of values of length `n_values`.

# Constructor

  - `Hypothesis(label,n_values)`: If a `content` vector is not supplied, it will be created.

"""
Base.@kwdef mutable struct Hypothesis <: HypothesisGeneration
    label
    n_values
    content
    # Optional constructor for null content
    Hypothesis(label,n_values) = new(label,n_values,generate_item(n_values))
end


"""
    Trace

Holds information about a single degraded trace within a HyGene simulation. Traces are
degraded using the `decay` parameter set in `HyGeneModel`.

# Arguments

  - `label`: Text label. (Mostly for house-keeping purposes.)
  - `n_values`: Length of feature mini-vectors.
  - `n_contexts`: Number of mini-vectors within a single item.
  - `last_active`: Trial in which this item was last retrieved. Not useful for base HyGene, but could be useful later on.
  - `content`: Vector (or mini-vectors if `n_features > 1`) of values representing the item.

"""
Base.@kwdef mutable struct Trace <: HypothesisGeneration
    # Static descriptors
    label
    n_values
    n_contexts
    # Variable descriptors
    last_active
    # Variable entities
    content
end