cd(@__DIR__)
using Pkg
# use package environment
Pkg.activate("")
using BenchmarkTools
Pkg.activate("..")
using HyGene
using Random

Random.seed!(8741)
n_features = 200
n_reps = fill(50, 6)
focal_prototype = make_traces(n_features)
alt_prototypes = make_traces(n_features, 5)
semantic_memory = hcat(focal_prototype, alt_prototypes)
episodic_memory = replicate_traces(semantic_memory, n_reps, 0.85)
probe = replicate_trace(focal_prototype[1:100], 0.85)

model = HyGeneModel(;
    # maximum consecutive retrieval failures 
    t_max = 5,
    # working memory capacity 
    κ = 4,
    # encoding fidelity
    ρ = 0.85,
    # retrieval threshold
    τ = 0.0,
    # indices for data components 
    data_map = (; data = 1:100),
    # indices for hypothesis components 
    hypothesis_map = (; hypothesis = 101:200),
    # episodic memory traces 
    episodic_memory,
    # semantic memory traces
    semantic_memory
)

@show judge_hypothesis(model, probe, 1)
@show model.working_memory;

@benchmark judge_hypothesis($model, $probe, $1)

@benchmark replicate_traces($semantic_memory, $n_reps, $0.85)
