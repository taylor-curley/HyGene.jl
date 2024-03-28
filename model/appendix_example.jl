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

# judge the probability of the hypothesis H1 given data represented by the probe 
@show judge_hypothesis(model, probe, 1)
@show model.working_memory;
