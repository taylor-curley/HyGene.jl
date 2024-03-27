abstract type AbstractHyGeneModel end

"""
mutable struct HyGeneModel{
    EM <: AbstractArray,
    SM <: AbstractArray,
    WM <: AbstractArray{<:Integer},
    T <: Real,
    NPD <: NamedTuple,
    NPH <: NamedTuple
} <: AbstractHyGeneModel

A standard HyGene model for predicting judgments of hypothesis probabilities. 

# Fields 

- `t_max::Int`: maximum consecutive retrieval failures 
- `κ::Int`: maximum hypotheses considered in working memory
- `ρ::T`: encoding fidelity 
- `τ::T`: retrieval threshold 
- `data_map::NPD`: a `NamedTuple` which maps data components (or minivectors) to indices of a trace
- `hypothesis_map::NPH`: a `NamedTuple` which maps hypothesis components (or minivectors) to indices of a trace
- `episodic_memory::EM`: an array representing epsodic memory in which rows correspond to features and columns correspond to traces
- `semantic_memory::SM`:  an array representing semantic memory in which rows correspond to features and columns correspond to traces
- `working_memory::WM`: a vector containing indices of hypotheses activate in working memory. Maximum size is determined by parameter `κ`
"""
mutable struct HyGeneModel{
    EM <: AbstractArray,
    SM <: AbstractArray,
    WM <: AbstractArray{<:Integer},
    T <: Real,
    NPD <: NamedTuple,
    NPH <: NamedTuple
} <: AbstractHyGeneModel
    t_max::Int
    κ::Int
    ρ::T
    τ::T
    data_map::NPD
    hypothesis_map::NPH
    episodic_memory::EM
    semantic_memory::SM
    working_memory::WM
end

function HyGeneModel(;
    t_max = 5,
    κ = 4,
    ρ = 0.85,
    τ = 0.21,
    data_map,
    hypothesis_map,
    episodic_memory,
    semantic_memory,
    working_memory = Int[]
)
    return HyGeneModel(
        t_max,
        κ,
        ρ,
        τ,
        data_map,
        hypothesis_map,
        episodic_memory,
        semantic_memory,
        working_memory
    )
end
