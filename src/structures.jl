"""
    HypothesisGeneration <: ContinuousUnivariateDistribution

An abstract type representing all structures related to HyGene simulations. 
Mostly useful when using custom samplers, such as with `Turing.jl`, but also
for preventing excessive function re-definition for different sub-types/structures.

# Type Hierarchy

```
HypothesisGeneration
├─ MemoryStore
│  ├─ LongTermMemory
│  ├─ SemanticMemory
│  └─ SetofContenders
├─ Model
│  └─ HyGeneModel
└─ Trace
   ├─ Information
   │  └─ Context
   ├─ MemoryTrace
   └─ ObsTrace
```

"""
abstract type HypothesisGeneration <: ContinuousUnivariateDistribution end


"""
    Model <: HypothesisGeneration

An abstract type representing large simulation objects. The base code only includes one
`Model` type ([`HyGeneModel`](@ref)), but users can create new objects for bespoke simulations.

# Subtypes

  - [`HyGeneModel`](@ref)

"""
abstract type Model <: HypothesisGeneration end


"""
    MemoryStore <: HypothesisGeneration

An abstract type representing all memory stores.

# Subtypes

  - [`LongTermMemory`](@ref)
  - [`SemanticMemory`](@ref)
  - [`SetofContenders`](@ref)

"""
abstract type MemoryStore <: HypothesisGeneration end


"""
    Trace <: MemoryStore

An abstract type representing chunks of information held in [`MemoryStore`](@ref) objects.

# Subtypes

  - [`MemoryTrace`](@ref)
  - [`ObsTrace`](@ref)
  
"""
abstract type Trace <: HypothesisGeneration end


"""
    Information <: Trace

An abstract type representing information held within chuncks in [`MemoryStore`](@ref) objects.
There is only one subtype in this abstract class, but more can be created for bespoke simulations.

# Subtypes

  - [`Context`](@ref)

"""
abstract type Information <: Trace end


"""
    Context <: Information

Container used to hold information about a single "context", generally defined. `Context`
structures are mutable and hold one numeric vector. They are considered to be the lowest
level of information in the `HypothesisGeneration` environment.

# Parameters

  - `label::Symbol`: Symbol indicating the overall type of `Context`. In simulations, similar `Context` structures will have the same `label`.
  - `description::String`: Optional description to help differentiate `Context` objects with similar labels.
  - `n_features::Integer`: Length of feature vector.
  - `contents::AbstractArray`: A single array of length `n_features`.
  - `A_i::Float64`: Optional value reflecting activation of the context. Typically unused during simulations.

# Constructor

  - [`Context(label::Symbol, description::String, n_features::Integer)`](@ref)
"""
@with_kw mutable struct Context <: Information
    label::Symbol
    description::String
    n_features::Integer
    contents::AbstractArray
    A_i::Float64
end


"""
    Context(label::Symbol, description::String, n_features::Integer)

Constructor for a new [`Context`](@ref) object. Generates a feature vector with values sampled from `[-1.0, 0.0, 1.0]`.

# Arguments

  - `label::Symbol`: Symbol indicating the overall type of [`Context`](@ref). In simulations, similar [`Context`](@ref) structures will have the same `label`.
  - `description::String`: Optional description to help differentiate [`Context`](@ref) objects with similar labels.
  - `n_features::Integer`: Length of resulting feature vector.

# Returns

  - `Context` object with randomly-generated feature vector of length `n_features`.

"""
function Context(label::Symbol, description::String, n_features::Integer)
    feature_vec = create_vec(n_features)
    return Context(label, description, n_features, feature_vec, 0.0)
end

"""
    Hypothesis
"""
@with_kw mutable struct Hypothesis <: Information
  label::Symbol
  description::String
  n_features::Integer
  contents::AbstractArray
  A_i::Float64
end
function Hypothesis(label::Symbol, description::String, n_features::Integer)
  feature_vec = create_vec(n_features)
  return Hypothesis(label, description, n_features, feature_vec, 0.0)
end


"""
    MemoryTrace <: Trace

Container for objects that populate `MemoryStore` objects. They consist of 1 or more [`Context`](@ref)
objects. These structures are analogous to information and memories that can potentially degrade
across time. [`LongTermMemory`](@ref) stores typically contain multiple permutations of specific `MemoryTrace`
objects while [`SemanticMemory`](@ref) holds single `MemoryTrace` objects for each type of observation.
All parameters of the `MemoryTrace` object are mutable.

# Parameters

  - `label::Symbol`: Symbol indicating the unique type of observation/trace.
  - `description::String`: Optional description to help differentiate traces with similar labels.
  - `n_contexts::Integer`: Number of unique types of [`Context`](@ref) objects. This is determined by finding unique `label` values across all `Context` objects.
  - `contexts::Vector{Symbol}`: List of unique types of [`Context`](@ref) objects.
  - `contents::Vector{<:Information}`: Contents of the memory store, i.e., 1 or more [`Context`](@ref) objects.
  - `A_i::Float64`: Activation value. Intitially set to `0.0` and then updated using the [`bounce_memory!`](@ref) function.

# Constructor

  - [`MemoryTrace(label::Symbol, description::String, info_vec::Vector{<:Information})`](@ref)

"""
@with_kw mutable struct MemoryTrace <: Trace
    label::Symbol
    description::String
    n_contexts::Integer
    contexts::Vector{Symbol}
    contents::Vector{<:Information}
    A_i::Float64
end


"""
    MemoryTrace(label::Symbol, description::String, info_vec::Vector{<:Information})

Constructor for a [`MemoryTrace`]@(def) object.

# Arguments

  - `label::Symbol`: Symbol indicating the unique type of observation/trace.
  - `description::String`: Optional description to help differentiate traces with similar labels.
  - `info_vec::Vector{<:Information}`: List of [`Information`](@ref)/[`Context`](@ref) objects to be written to the resulting [`MemoryTrace`](@ref) object.

# Returns

  - [`MemoryTrace`](@ref) object.

"""
function MemoryTrace(label::Symbol, description::String, info_vec::Vector{<:Information})
    contexts = Vector([info.label for info in info_vec])
    return MemoryTrace(
        label,
        description,
        length(info_vec),
        contexts,
        deepcopy(info_vec),
        0.0,
    )
end


"""
    ObsTrace <: Trace

Structurally similar to the [`MemoryTrace`](@ref) object type, but represents an observation made in the
"real world" that is then compared to a [`MemoryStore`](@ref) object. Typically unneeded, but can be 
useful for operations in parametric functions that differentiate memory traces and traces derived
from outward observations.

# Parameters

  - `label::Symbol`: Symbol indicating the unique type of observation/trace.
  - `description::String`: Optional description to help differentiate traces with similar labels.
  - `n_contexts::Integer`: Number of unique types of [`Context`](@ref) objects. This is determined by finding unique `label` values across all [`Context`](@ref) objects.
  - `contexts::Vector{Symbol}`: List of unique types of [`Context`](@ref) objects.
  - `contents::Vector{<:Information}`: Contents of the memory store, i.e., 1 or more [`Context`](@ref) objects.
  - `A_i::Float64`: Activation value. Typically unused.

# Constructor

  - [`ObsTrace(label::Symbol, description::String, info_vec::Vector{<:Information})`](@ref)

"""
@with_kw mutable struct ObsTrace <: Trace
    label::Symbol
    description::String
    n_contexts::Integer
    contexts::Vector{Symbol}
    contents::Vector{<:Information}
    A_i::Float64
end


"""
    ObsTrace(label::Symbol, description::String, info_vec::Vector{<:Information})

Constructor for a [`ObsTrace`](@ref) object.

# Arguments

  - `label::Symbol`: Symbol indicating the unique type of observation/trace.
  - `description::String`: Optional description to help differentiate traces with similar labels.
  - `info_vec::Vector{<:Information}`: List of [`Information`](@ref)/[`Context`](@ref) objects to be written to the resulting [`ObsTrace`](@ref) object.

# Returns

  - [`ObsTrace`](@ref) object.

"""
function ObsTrace(label::Symbol, description::String, info_vec::Vector{<:Information})
    contexts = Vector([info.label for info in info_vec])
    return ObsTrace(label, description, length(info_vec), contexts, info_vec, 0.0)
end


"""
    LongTermMemory <: MemoryStore

A long-term memory store for [`Trace`](@ref) objects. Contents in [`LongTermMemory`](@ref) are potentially subject 
to degradation ([`trace_decay!`](@ref)). [`LongTermMemory`](@ref) is key in echo content calculations and posterior
probability estimates resulting from the SoC. Items in this store are used to create exemplars
in [`SemanticMemory`](@ref).

# Parameters

  - `n_items::Integer`: The number of [`Trace`](@ref) items in storage.
  - `A_c::Float64`: Activation threshold for conditional retrieval/probability. Used during the [`echo_content`](@ref) and [`cond_echo_intensity`](@ref) operations.
  - `unique_contexts::Vector{Symbol}`: List of unique symbols extracted from [`Trace`](@ref) objects.
  - `contents::Vector{<:Trace}`: List of traces in long-term memory.

# Constructor

  - [`LongTermMemory(trace_vec::Vector{<:Trace}, A_c::Float64)`](@ref)

"""
@with_kw mutable struct LongTermMemory <: MemoryStore
    n_items::Integer
    A_c::Float64
    unique_contexts::Vector{Symbol}
    contents::Vector{<:Trace}
end


"""
    LongTermMemory(trace_vec::Vector{<:Trace}, A_c::Float64)

Constructor for a [`LongTermMemory`](@ref) object.

# Arguments

  - `trace_vec::Vector{<:Trace}`: List of traces to be written to long-term memory.
  - `A_c::Float64`: Activation threshold. Used during the [`echo_content`](@ref) and [`cond_echo_intensity`](@ref) operations.

# Returns

  - [`LongTermMemory`](@ref) object.

"""
function LongTermMemory(trace_vec::Vector{<:Trace}, A_c::Float64)
    contexts = Vector{Symbol}()
    for trace in trace_vec
        for context in trace.contexts
            push!(contexts, deepcopy(context))
        end
    end
    return LongTermMemory(length(trace_vec), A_c, unique(contexts), trace_vec)
end


"""
    SemanticMemory <: MemoryStore

A memory store consisting of [`Trace`](@ref) objects. All of the items in SemanticMemory are exemplars derived from
multiple observations of individual events in [`LongTermMemory`](@ref). For example, in a [`LongTermMemory`](@ref) store with
10 separate traces with label `:event_a` and 10 separate traces with label `:event_b`, SemanticMemory` would
be populated with two separate traces for `:event_a` and `:event_b` with their item vectors being the means
of the contents of the 10 separate traces for each event.

# Parameters

  - `n_items::Integer`: The number of unique exemplars, or semantic memory traces.
  - `content_labels::Vector{Symbol}`: List of the labels for the unique exemplars.
  - `contents::Vector{<:Trace}`: List of unique exemplars in semantic memory.
  - `A_c::Float64`: Activation threshold for condition retrieval/probability. Not typically used in these simulations.

# Constructor

  - [`SemanticMemory(ltm::LongTermMemory)`](@ref)

"""
@with_kw mutable struct SemanticMemory <: MemoryStore
    n_items::Integer
    content_labels::Vector{Symbol}
    contents::Vector{<:Trace}
    A_c::Float64
end


"""
    SemanticMemory(ltm::LongTermMemory)
    SemanticMemory(exemplars::Vector{<:Trace})

Constructor for the [`SemanticMemory`](@ref) structure class. If there are only contents
in long-term memory, then this operation will create averaged feature vectors. If prototypes
are given, then [`SemanticMemory`](@ref) will be populated with those.

# Arguments

  - `ltm::LongTermMemory`: Memory store from which exemplar traces are to be extracted.

# Returns

  - [`SemanticMemory`](@ref) object.
"""
function SemanticMemory(ltm::LongTermMemory)
    # Determine the number of unique event types
    unique_event_tags = unique([item.label for item in ltm.contents])
    # Pre-allocate trace vector
    semantic_trace_vector = Vector{MemoryTrace}()
    for tag in unique_event_tags
        # Subset list for a unique event
        event_traces = filter(n -> n.label == tag, ltm.contents)
        event_contexts = event_traces[1].contexts
        # Pre-allocate vector and new trace
        semantic_trace = deepcopy(event_traces[1])
        semantic_trace_contexts = Vector{Context}()
        for context in event_contexts
            trace_contexts = Vector([
                filter(n -> n.label == context, trace.contents)[1] for trace in event_traces
            ])
            copy_context = deepcopy(trace_contexts[1])
            context_arrays = Vector([context.contents for context in trace_contexts])
            mean_contents = sum(context_arrays) ./ length(context_arrays)
            copy_context.contents = mean_contents
            push!(semantic_trace_contexts, copy_context)
        end
        semantic_trace.contents = semantic_trace_contexts
        standardize_trace!(semantic_trace)
        push!(semantic_trace_vector, semantic_trace)
    end
    return SemanticMemory(
        length(semantic_trace_vector),
        unique_event_tags,
        semantic_trace_vector,
        copy(ltm.A_c),
    )
end
function SemanticMemory(exemplars::Vector{<:Trace})
    event_tags = [item.label for item in exemplars]
    return SemanticMemory(length(exemplars), event_tags, exemplars, 0.0)
end

"""
    SetofContenders <: MemoryStore

An analog to the "Working Memory" concept in Thomas et al.'s (2008) [^1] original HyGene
paper. The `SetofContenders` object is used to hold above-threshold traces from [`SemanticMemory`](@ref)
when an outward observation ([`ObsTrace`](@ref)) is compared against its contents. The
contents of the `SetofContenders` is then used to compute the probability that a given
semantic representation is the correct exemplar, given the probe ([`cond_echo_intensity`](@ref)).

# Parameters

  - `n_contenders::Integer`:
  - `contenders::Vector{<:MemoryTrace}`:
  - `act_min::Float64`:
  - `t_max::Integer`:
  - `t::Integer`:

# Constructor

  - [`SetofContenders(t_max::Integer)`](@ref)

"""
@with_kw mutable struct SetofContenders <: MemoryStore
    n_contenders::Integer
    contenders::Vector{<:MemoryTrace}
    act_min::Float64
    t_max::Integer
    t::Integer
end


"""
    SetofContenders(contenders::Vector{<:MemoryTrace}, t_max::Integer)
    SetofContenders(t_max::Integer)

Constructor function for the `SetofContenders` object. `SetofContenders` objects are
typically instantiated with no contents.

# Arguments

  - `contenders::Vector{<:MemoryTrace}`: Vector of memory traces to populate the SoC.
  - `t_max::Integer`: Maximum number of retrieval failures.

# Returns

  - [`SetofContenders`](@ref) object.

"""
function SetofContenders(contenders::Vector{<:MemoryTrace}, t_max::Integer)
    return SetofContenders(length(contenders), contenders, 0.0, t_max, t)
end
function SetofContenders(t_max::Integer)
    return SetofContenders(0, Vector{MemoryTrace}(), 0.0, t_max, 0)
end


"""
    HyGeneModel <: Model

Object holding [`HypothesisGeneration`](@ref) simulation parameters, variables, and outcomes. The top-level
operations that govern the `HyGeneModel` object are located in `sim_controllers.jl`.

# Parameters

  - `A_c::Float64`: Activation threshold in [`LongTermMemory`](@ref).
  - `t_max::Integer`: Maximum number of retrieval failures during [`SetofContenders`](@ref) operations.
  - `decay::Float64`: Degree of trace decay for [`LongTermMemory`](@ref) contents.
  - `focal_similarity::Float64`: Degree of similarity between the focal and alternative hypotheses.
  - `encoding_fidelity::Float64`: Degree of change between an observation in the environment and a new [`MemoryTrace`](@ref) object.
  - `n_features::Integer`: Length of context mini-vectors. `10` is recommended.
  - `n_unique_contexts::Integer`: Number of unique [`Context`](@ref) types.
  - `n_unique_events::Integer`: Number of unique [`MemoryTrace`](@ref) types in [`LongTermMemory`](@ref) and length of [`SemanticMemory`](@ref) store contents.
  - `n_obs_per_event::Any`: Number of unqiue observations per unique [`MemoryTrace`](@ref) types in [`LongTermMemory`](@ref).
  - `contexts::Vector{Symbol}`: List of labels of unique types of [`Context`](@ref) objects.
  - `hypotheses::Vector{Symbol}`: List of labels of unique types of [`MemoryTrace`](@ref) objects.
  - `len_ltm::Integer`: Number of items in the [`LongTermMemory`](@ref) store. Should be `n_unique_events * n_obs_per_event`.
  - `len_sm::Integer`: Number of items in the [`SemanticMemory`](@ref) store. Should be `n_unique_events`.
  - `prototypes::Vector{<:Trace}`: List of hypothesis exemplars.
  - `long_term_memory::LongTermMemory`: A single [`LongTermMemory`](@ref) object.
  - `semantic_memory::SemanticMemory`: A single [`SemanticMemory`](@ref) object.
  - `working_memory::SetofContenders`: A single [`SetofContenders`](@ref) object.

"""
@with_kw mutable struct HyGeneModel <: Model
    # Simulation parameters
    A_c::Float64
    t_max::Integer
    focal_similarity::Float64
    encoding_fidelity::Float64
    n_features::Integer
    n_unique_contexts::Integer
    n_unique_events::Integer
    n_obs_per_event::Any
    # Descriptives
    contexts::Vector{Symbol}
    hypotheses::Vector{Symbol}
    len_ltm::Integer
    len_sm::Integer
    prototypes::Vector{<:Trace}
    # Memory stores
    long_term_memory::LongTermMemory
    semantic_memory::SemanticMemory
    working_memory::SetofContenders
end


"""
    HyGeneModel(context_labels::Vector{Symbol}, hypothesis_labels::Vector{Symbol},
                n_trace_vec::Vector{<:Number}, A_c::Number=0.2, t_max::Integer=10,
                n_features::Integer=15, focal_similarity::Float64=0.0, encoding_fidelity::Float64=0.75)
    HyGeneModel(n_contexts::Integer, n_hypotheses::Integer, n_obs_per_proto::Integer,
                A_c::Number=0.2, t_max::Integer=10, n_features::Integer=15,
                focal_similarity::Float64=0.0, encoding_fidelity::Float64=0.75)

Constructor functions for the [`HyGeneModel`](@ref) object class.
"""
function HyGeneModel(
    context_labels::Vector{Symbol},
    hypothesis_labels::Vector{Symbol},
    n_trace_vec::Vector{<:Number},
    A_c::Number = 0.2,
    t_max::Integer = 10,
    n_features::Integer = 15,
    focal_similarity::Float64 = 0.0,
    encoding_fidelity::Float64 = 0.75,
)

    # 1. Create prototypes
    prototypes =
        create_prototypes(context_labels, hypothesis_labels, n_features, focal_similarity)

    # 2. Create traces and populate memory stores
    traces = create_traces(prototypes, n_trace_vec, encoding_fidelity)
    ltm = LongTermMemory(traces, A_c)
    sm = SemanticMemory(ltm)
    wm = SetofContenders(t_max)

    # 3. Return model object
    return HyGeneModel(
        A_c,
        t_max,
        focal_similarity,
        encoding_fidelity,
        n_features,
        length(context_labels),
        length(hypothesis_labels),
        n_trace_vec,
        context_labels,
        hypothesis_labels,
        length(traces),
        length(sm.contents),
        prototypes,
        ltm,
        sm,
        wm,
    )
end
function HyGeneModel(
    n_contexts::Integer,
    n_hypotheses::Integer,
    n_obs_per_proto::Integer,
    A_c::Number = 0.2,
    t_max::Integer = 10,
    n_features::Integer = 15,
    focal_similarity::Float64 = 0.0,
    encoding_fidelity::Float64 = 0.75,
)
    # Create label and numeric vectors
    context_labels = create_labels(n_contexts, "context_")
    hypothesis_labels = create_labels(n_hypotheses, "hypothesis_")
    n_trace_vec = ones(n_hypotheses) * n_obs_per_proto
    # Pass to higher function
    HyGeneModel(
        context_labels,
        hypothesis_labels,
        n_trace_vec,
        A_c,
        t_max,
        n_features,
        focal_similarity,
        encoding_fidelity,
    )
end

function StatsBase.describe(hygene::HyGeneModel)
    out = [
        "A_c" hygene.A_c "Activation threshold \nin long-term memory."
        "t_max" hygene.t_max "Maximum number of \nretrieval failures."
    ]
    header = ["Param.", "Value", "Description"]

    return pretty_table(
        out,
        linebreaks = true,
        body_hlines = [1, 2],
        header = header,
        alignment = :l,
    )
end
