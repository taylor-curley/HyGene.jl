module HypothesisGeneration

# Import external libraries
using Distributions, Parameters, StatsBase, Documenter, DocumenterTools, DataFrames, PrettyTables

# Structure exports
export HyGene, Model, MemoryStore, Trace, Information, ObsTrace
export Context, MemoryTrace, LongTermMemory, SemanticMemory
export SetofContenders, HyGeneModel
include("./structures.jl")

# Utility exports
export create_vec, trace_replication, trace_activation
export trace_similarity, trace_decay!, bounce_memory!
export clear_content!, standardize_trace!, echo_content
export populate_soc!, cond_echo_intensity, soc_posterior_prob
export soc_winner
include("./utilities.jl")

# Simulation controller exports
export create_prototypes, create_traces, create_observation, create_labels
include("./sim_controller.jl")

end
