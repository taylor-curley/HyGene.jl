abstract type AbstractHyGeneModel end

mutable struct HyGeneModel{
    EM<:AbstractArray,
    SM<:AbstractArray,
    WM<:AbstractArray{<:Integer},
    T<:Real,
    NPD<:NamedTuple,
    NPH<:NamedTuple,
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
    working_memory = Int[],
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
        working_memory,
    )
end
