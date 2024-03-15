# Constructors

## Index

```@index
Pages = ["constrs.md"]
```

## Functions

```@docs
HyGeneModel(context_labels::Vector{Symbol}, hypothesis_labels::Vector{Symbol},
            n_trace_vec::Vector{<:Number}, A_c::Number=0.2, t_max::Integer=10,
            n_features::Integer=15, focal_similarity::Float64=0.0, encoding_fidelity::Float64=0.75)
Context(label::Symbol, description::String, n_features::Integer)
MemoryTrace(label::Symbol, description::String, info_vec::Vector{<:Information})
ObsTrace(label::Symbol, description::String, info_vec::Vector{<:Information})
LongTermMemory(trace_vec::Vector{<:Trace}, A_c::Float64)
SemanticMemory(ltm::LongTermMemory)
SetofContenders(contenders::Vector{<:MemoryTrace}, t_max::Integer)
```