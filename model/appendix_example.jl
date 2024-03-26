cd(@__DIR__)
using Pkg
# use package environment
Pkg.activate("..")
using HyGene
Pkg.activate("")
using StatsBase, StatsPlots, ProgressMeter, DataFrames, Plots

# Define traces
traces = [[0,   0,  -1, 1,  0,  1,  -1, 0,  0], # Trace1 (E1)
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
hypotheses = [[1,   -1, 0,  0,  1,  1,  -1,  1, 0], # Trace1 (E1)
              [1,   -1, 0,  0,  1,  1,  -1,  1, 0],
              [1,   -1, 0,  0,  1,  1,  -1,  1, 0],
              [1,   0,  0,  0,  1,  1,  -1,  1, 0],
              [1,   0,  0,  0,  1,  1,  -1,  1, 0],
              [1,   0,  0,  0,  1,  1,  -1,  1, 0],
              [1,   -1, 0,  0,  0,  1,  0,  1,  0],
              [0,   -1, 0,  0,  0,  1,  0,  1,  0],
              [1,   -1, 0,  0,  0,  1,  0,  1,  0],
              [1,   0,  0,  0,  0,  0,  -1, 1,  0]]

# Define probe
probe = [0, 1,  -1, 1,  0,  1,  -1, 1,  0]

# Trace activation of first trace to probe should be 0.296
trace_1_act = trace_activation(probe, traces[1])

# Get activations for all traces
trace_acts = trace_activation(probe, traces)

# Mask values below activation threshold and subset traces
mask = trace_acts .> 0.296
trace_subset = traces[mask]
trace_acts_subset = trace_acts[mask]

# Calculate conditional echo content for data (not hypothesis)
cond_echo_content = trace_subset .* trace_acts_subset
cond_echo_content = sum(cond_echo_content)

# Generate unspecified probe by normalizing vector
unspec_probe = cond_echo_content ./ maximum(cond_echo_content)
