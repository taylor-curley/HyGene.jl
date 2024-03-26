cd(@__DIR__)
using Pkg
# use package environment
Pkg.activate("..")
using HyGene
Pkg.activate("")
using StatsBase, StatsPlots, ProgressMeter, DataFrames, Plots

# Define traces
traces =       [[0,   0,  -1, 1,  0,  1,  -1, 0,  0], # Trace1 (E1)
                [0,   1,  -1, 1,  0,  1,  -1, 1,  0], # Trace2 (E1)
                [0,   1,  -1, 0,  0,  1,  -1, 1,  0], # Trace3 (E1)
                [0,   1,  -1, 1,  0,  1,  -1, 1,  0], # Trace4 (E1)
                [0,   1,  -1, 1,  0,  1,  -1, 1,  0], # Trace5 (E1)
                [0,   0,  -1, 1,  0,  1,  -1, 1,  0], # Trace6 (E1)
                [-1,  0,  -1, 0,  -1, 1,  1,  0,  0], # Trace7 (E2)
                [-1,  0,  -1, 1,  -1, 1,  1,  0,  0], # Trace8 (E2)
                [-1,  0,  0,  1,  -1, 1,  1,  1,  0], # Trace9 (E2)
                [1,   0,  -1, -1, -1, 1,  -1, 0,  0]] # Trace10 (E3)

# Define hypotheses
hypotheses =   [[1,   -1, 0,  0,  1,  1,  -1,  1, 0], # Trace1 (E1)
                [1,   -1, 0,  0,  1,  1,  -1,  1, 0], # Trace2 (E1)
                [1,   -1, 0,  0,  1,  1,  -1,  1, 0], # Trace3 (E1)
                [1,   0,  0,  0,  1,  1,  -1,  1, 0], # Trace4 (E1)
                [1,   0,  0,  0,  1,  1,  -1,  1, 0], # Trace5 (E1)
                [0,   0,  0,  0,  1,  1,  -1,  1, 0], # Trace6 (E1)
                [1,   -1, 0,  0,  0,  1,  0,  1,  0], # Trace7 (E2)
                [0,   -1, 0,  0,  0,  1,  0,  1,  0], # Trace8 (E2)
                [1,   -1, 0,  0,  0,  1,  0,  1,  0], # Trace9 (E2)
                [1,   0,  0,  0,  0,  0,  -1, 1,  0]] # Trace10 (E3)

# Define probe
probe = [0, 1,  -1, 1,  0,  1,  -1, 1,  0]

# Define semantic traces
sem_traces = [[0,   1,  -1, 1,  0,  1,  -1, 1,  0],
              [-1,  0,  -1, 1,  -1, 1,  1,  1,  0],
              [1,   0,  -1, -1, 1,  1,  -1, 0,  0]]

# Define semantic hypotheses
sem_hyp =   [[1,    -1, 0,  0,  1,  1,  -1, 1,  0],
             [1,    -1, 0,  0,  0,  1,  0,  1,  0],
             [1,    0,  0,  0,  0,  1,  -1, 1,  0]]

# Trace activation of first trace to probe should be 0.296
trace_1_act = trace_activation(probe, traces[1])

# Get activations for all trace
trace_acts = trace_activation(probe, traces)

# Mask values below activation threshold and subset
mask = trace_acts .> 0.216
trace_subset = traces[mask]
trace_acts_subset = trace_acts[mask]
hyp_subset = hypotheses[mask]

# Calculate conditional echo content for data
cond_echo_cont_data = trace_subset .* trace_acts_subset
cond_echo_cont_data = sum(cond_echo_cont_data)

# Calculate conditional echo content for hypotheses
cond_echo_cont_hyp = hyp_subset .* trace_acts_subset
cond_echo_cont_hyp = sum(cond_echo_cont_hyp)

# Generate unspecified probe by normalizing vector
unspec_probe = vcat(cond_echo_cont_data,cond_echo_cont_hyp)
unspec_probe ./= maximum(unspec_probe)

# Calculate semantic activation to unspecified probe
semantic_acts = [trace_activation(unspec_probe, vcat(x,y)) for (x,y) in zip(sem_traces,sem_hyp)]

# Normalize semantic activations
norm_sem_acts = semantic_acts / sum(semantic_acts)

# From above-threshold episodic traces, compare hypothesis components 
# for episodic traces and first semantic trace
cond_echo_int = trace_activation(sem_hyp[1], hyp_subset)
cond_echo_int = mean(cond_echo_int)