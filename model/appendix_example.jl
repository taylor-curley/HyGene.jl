cd(@__DIR__)
using Pkg
# use package environment
Pkg.activate("..")
using HyGene

# memory probe corresponding to the data component 
probe = [0, 1, -1, 1, 0, 1, -1, 1, 0]
# episodic memory: rows are features, and columns are traces
# rows 1:9 correspond to data component
# rows 10:18 correspond to the  hypothesis component
episodic_memory = [
    0 0 0 0 0 0 -1 -1 -1 1
    0 1 1 1 1 0 0 0 0 0
    -1 -1 -1 -1 -1 -1 -1 -1 0 -1
    1 1 0 1 1 1 0 1 1 -1
    0 0 0 0 0 0 -1 -1 -1 1
    1 1 1 1 1 1 1 1 1 1
    -1 -1 -1 -1 -1 -1 1 1 1 -1
    0 1 1 1 1 1 0 0 1 0
    0 0 0 0 0 0 0 0 0 0
    1 1 1 1 1 0 1 0 1 1
    -1 -1 -1 0 0 0 -1 -1 -1 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    1 1 1 1 1 1 0 0 0 0
    1 1 1 1 1 1 1 1 1 0
    -1 -1 -1 -1 -1 -1 0 0 0 -1
    1 1 1 1 1 1 1 1 1 1
    0 0 0 0 0 0 0 0 0 0
]
semantic_memory = [
    0 -1 1
    1 0 0
    -1 -1 -1
    1 1 -1
    0 -1 1
    1 1 1
    -1 1 -1
    1 1 0
    0 0 0
    1 1 1
    -1 -1 0
    0 0 0
    0 0 0
    1 0 0
    1 1 1
    -1 0 -1
    1 1 1
    0 0 0
]

model = HyGeneModel(;
    # maximum consecutive retrieval failures 
    t_max = 5,
    # working memory capacity 
    κ = 4,
    # encoding fidelity
    ρ = 0.85,
    # retrieval threshold
    τ = 0.21,
    # indices for data components 
    data_map = (; data = 1:9),
    # indices for hypothesis components 
    hypothesis_map = (; hypothesis = 10:18),
    # episodic memory traces 
    episodic_memory,
    # semantic memory traces
    semantic_memory
)

# judge the probability of the hypotheses given data represented by the probe 
@show judge_hypotheses(model, probe)
@show model.working_memory;

using Random

Random.seed!(8741)
n_features = 200
n_reps = fill(50, 6)
focal_protype = make_traces(n_features)
alt_prototypes = make_traces(n_features, 5)
semantic_memory = hcat(focal_protype, alt_prototypes)
episodic_memory = replicate_traces(semantic_memory, n_reps, 0.85)
probe = replicate_trace(focal_protype[1:100], 0.85)

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

@show judge_hypotheses(model, probe)
@show model.working_memory;
