cd(@__DIR__)
using Pkg
# use package environment
Pkg.activate("..")
using HyGene
using Revise

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
    data_map = (;d1=1:9),
    hypothesis_map = (;hypothesis=10:18),
    episodic_memory,
    semantic_memory
)

# judge the probability of the hypotheses given data represented by the probe 
@show judge_hypotheses(model, probe)
@show model.working_memory;

# @benchmark judge_hypotheses($model, $probe)

# BenchmarkTools.Trial: 10000 samples with 10 evaluations.
#  Range (min … max):  1.406 μs … 384.712 μs  ┊ GC (min … max): 0.00% … 98.74%
#  Time  (median):     1.496 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   1.653 μs ±   5.057 μs  ┊ GC (mean ± σ):  4.27% ±  1.40%

#   ▄▇█▇▅▄▃▁                        ▁                           ▂
#   ██████████▆▆▅▁▅▄▁▄▄▃▃▃▄▁▄▃▄▆███████▇▇▆▆▇▅▃▅▆▆▄▃▅▁▄▄▅▅▃▅▅▆▆▆ █
#   1.41 μs      Histogram: log(frequency) by time      3.47 μs <

#  Memory estimate: 1.09 KiB, allocs estimate: 12.