module HyGene

using StatsBase

export AbstractHyGeneModel
export HyGeneModel

export compute_activations
export compute_cond_echo_content
export compute_cond_echo_intensity
export compute_cond_echo_intensities
export create_unspecified_probe
export judge_hypothesis
export judge_posterior
export make_traces
export populate_working_memory!
export replicate_trace
export replicate_traces

include("structs.jl")
include("functions.jl")
include("utilities.jl")
end
