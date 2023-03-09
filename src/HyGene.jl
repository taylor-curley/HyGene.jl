module HyGene

# Useful packages
using StatsBase, Plots, StatsPlots, Distributions, DataFrames, Revise

# Structure exports
export HypothesisGeneration, HyGeneModel, Observation, Context, Hypothesis, Trace
include("./structures.jl")

# Simulation controller exports
export generate_item, trace_decay, trace_decay!, trace_similarity
export trace_activation, echo_intensity, conditional_echo_intensity
export obs_to_trace, trace_replicator, echo_content, get_contenders!
export generate_observation
include("./sim_controllers.jl")

# Export all the random functions 
export block_header, line_header
include("./misc.jl")

end
