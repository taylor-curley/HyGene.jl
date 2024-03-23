
abstract type AbstractHygene end

mutable struct Hygene{
    EM<:AbstractArray,
    SM<:AbstractArray,
    WM<:AbstractArray{<:Integer},
    T<:Real,
    F,
    R,
} <: AbstractHygene

    t_max::Int
    focal_similarity::T
    encoding_fidelity::T
    threshold::T
    data_map::Dict{F,R}
    hypothesis_map::Dict{F,R}
    episodic_memory::EM
    semantic_memory::SM
    working_memory::WM
end

function Hygene(;
    t_max,
    focal_similarity,
    encoding_fidelity,
    threshold,
    data_map,
    hypothesis_map,
    episodic_memory,
    semantic_memory,
    working_memory,
)

    return Hygene(
        t_max,
        focal_similarity,
        encoding_fidelity,
        threshold,
        data_map,
        hypothesis_map,
        episodic_memory,
        semantic_memory,
        working_memory,
    )
end
