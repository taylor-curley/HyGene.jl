cd(@__DIR__)
using Pkg
# use package environment
Pkg.activate("..")
using HyGene
Pkg.activate("")
using StatsBase, StatsPlots, ProgressMeter, DataFrames, Plots

# SIMULATION 1: THOMAS ET AL. (2008)

## SIMULATION 1A: CONSTRAINED MODEL

# General parameters
A_c = 0.215         # Activation threshold in LTM
t_max = 10          # Maximum retrieval failures
n_sims = 100        # Number of simulations per parameter combination
n_trials = 10       # Number of trials per simulation
n_features = 15     # Length of context mini-vectors

# Define 10 contexts with unique labels. Contexts should not be similar
# to each other.
context_labels = [
    :disease,
    :symptom_1,
    :symptom_2,
    :symptom_3,
    :symptom_4,
    :symptom_5,
    :symptom_6,
    :symptom_7,
    :symptom_8,
    :symptom_9
]

# Define 8 hypotheses with unique labels. The alt hypotheses should have
# a similarity to the focal hypothesis given by a float value.
hypothesis_labels = [:focal, :alt_1, :alt_2, :alt_3, :alt_4, :alt_5, :alt_6, :alt_7]
focal_sim = collect(0.0:0.1:1.0)

# Define trace representation in the "Focal prevalent" condition.
n_traces = fill(10, 8)
n_traces[1] = 70

# Define the relative proportion of features shared between LTM traces and their
# prototypes.
encode_fidelity = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# Define the similarity between a real-world observation and a given hypothesis
# exemplar.
d_obs_sim = 0.85

# Instantiate a blank dataframe
sim_1_results = DataFrame(
    A_c = Number[],
    t_max = Integer[],
    n_features = Integer[],
    d_obs_sim = Number[],
    focal_sim = Number[],
    encode_fidelity = Number[],
    probe_type = Symbol[],
    n_contenders = Integer[],
    recalled_type = Any[],
    ACC = Integer[]
)

# Iterate through parameter sets
p = Progress(length(focal_sim) * length(encode_fidelity) * n_sims * n_trials)
for S in focal_sim
    for L in encode_fidelity
        for i = 1:n_sims
            # Create simulation object
            sim_mod = HyGeneModel(
                context_labels,
                hypothesis_labels,
                n_traces,
                A_c,
                t_max,
                n_features,
                S,
                L
            )
            for j = 1:n_trials
                # Randomly select hypothesis and generate trace
                # Select focal observation with 0.5 probability
                if rand() > 0.5
                    obs = trace_replication(rand(sim_mod.prototypes[2:end]), d_obs_sim)
                else
                    obs = trace_replication(sim_mod.prototypes[1], d_obs_sim)
                end

                # Generate hologram from observation against long-term memory
                echo = echo_content(obs, sim_mod.long_term_memory)

                # Populate the SoC and get the posterior predictive values
                populate_soc!(echo, sim_mod)
                n_contenders = 0
                if sim_mod.working_memory.n_contenders > 0
                    n_contenders = sim_mod.working_memory.n_contenders
                    if n_contenders == 1
                        post_prob = 1.0
                        winner = sim_mod.working_memory.contenders[1]
                    else
                        post_prob = soc_posterior_prob(
                            sim_mod.working_memory,
                            sim_mod.long_term_memory
                        )
                        winner = sim_mod.working_memory.contenders[argmax(
                            getfield.(sim_mod.working_memory.contenders, :A_i),
                        )]
                    end
                    winner_label = winner.label
                    acc = (obs.label == winner.label) * 1
                else
                    winner = NaN
                    winner_label = NaN
                    acc = 0
                end

                # Push data to DF
                out = [
                    A_c,
                    t_max,
                    n_features,
                    d_obs_sim,
                    S,
                    L,
                    obs.label,
                    n_contenders,
                    winner_label,
                    acc
                ]
                push!(sim_1_results, out)
                next!(p)
            end
        end
    end
end
# Recode probe type
sim_1_results.probe_type_2 .= :focal
sim_1_results.probe_type_2[sim_1_results.probe_type .!= :focal] .= :alt

# Calculate average n_contenders by probe type, S, and L
sim_1_avg = combine(
    groupby(sim_1_results, [:probe_type_2, :focal_sim, :encode_fidelity]),
    [:n_contenders, :ACC] .=> mean
)

# Average accuracy
lay = @layout [a b c d e f]
a = @df sim_1_avg[sim_1_avg.encode_fidelity .== 0.1, :] plot(
    :focal_sim,
    :ACC_mean,
    groups = :probe_type_2,
    lims = (0, 1),
    grid = false,
    title = "L = 0.1",
    left_margin = 0 * Plots.mm,
    right_margin = -6 * Plots.mm
)
b = @df sim_1_avg[sim_1_avg.encode_fidelity .== 0.3, :] plot(
    :focal_sim,
    :ACC_mean,
    groups = :probe_type_2,
    lims = (0, 1),
    grid = false,
    label = false,
    showaxis = :x,
    title = "L = 0.3",
    left_margin = 0 * Plots.mm,
    right_margin = -6 * Plots.mm
)
c = @df sim_1_avg[sim_1_avg.encode_fidelity .== 0.5, :] plot(
    :focal_sim,
    :ACC_mean,
    groups = :probe_type_2,
    lims = (0, 1),
    grid = false,
    label = false,
    showaxis = :x,
    title = "L = 0.5",
    left_margin = 0 * Plots.mm,
    right_margin = -6 * Plots.mm
)
d = @df sim_1_avg[sim_1_avg.encode_fidelity .== 0.7, :] plot(
    :focal_sim,
    :ACC_mean,
    groups = :probe_type_2,
    lims = (0, 1),
    grid = false,
    label = false,
    showaxis = :x,
    title = "L = 0.7",
    left_margin = 0 * Plots.mm,
    right_margin = -6 * Plots.mm
)
e = @df sim_1_avg[sim_1_avg.encode_fidelity .== 0.9, :] plot(
    :focal_sim,
    :ACC_mean,
    groups = :probe_type_2,
    lims = (0, 1),
    grid = false,
    label = false,
    showaxis = :x,
    title = "L = 0.9",
    left_margin = 0 * Plots.mm,
    right_margin = -6 * Plots.mm
)
f = @df sim_1_avg[sim_1_avg.encode_fidelity .== 1.0, :] plot(
    :focal_sim,
    :ACC_mean,
    groups = :probe_type_2,
    lims = (0, 1),
    grid = false,
    label = false,
    showaxis = :x,
    title = "L = 1.0",
    left_margin = 0 * Plots.mm
)
plot(
    a,
    b,
    c,
    d,
    e,
    f,
    layout = lay,
    size = (1_200, 400),
    plot_title = "Average Accuracy",
    xlabel = "S"
)

# Average SOC n
g = @df sim_1_avg[sim_1_avg.encode_fidelity .== 0.1, :] plot(
    :focal_sim,
    :n_contenders_mean,
    groups = :probe_type_2,
    ylims = (0, 3),
    grid = false,
    title = "L = 0.1",
    left_margin = 0 * Plots.mm,
    right_margin = -6 * Plots.mm
)
h = @df sim_1_avg[sim_1_avg.encode_fidelity .== 0.3, :] plot(
    :focal_sim,
    :n_contenders_mean,
    groups = :probe_type_2,
    ylims = (0, 3),
    grid = false,
    label = false,
    showaxis = :x,
    title = "L = 0.3",
    left_margin = 0 * Plots.mm,
    right_margin = -6 * Plots.mm
)
i = @df sim_1_avg[sim_1_avg.encode_fidelity .== 0.5, :] plot(
    :focal_sim,
    :n_contenders_mean,
    groups = :probe_type_2,
    ylims = (0, 3),
    grid = false,
    label = false,
    showaxis = :x,
    title = "L = 0.5",
    left_margin = 0 * Plots.mm,
    right_margin = -6 * Plots.mm
)
j = @df sim_1_avg[sim_1_avg.encode_fidelity .== 0.7, :] plot(
    :focal_sim,
    :n_contenders_mean,
    groups = :probe_type_2,
    ylims = (0, 3),
    grid = false,
    label = false,
    showaxis = :x,
    title = "L = 0.7",
    left_margin = 0 * Plots.mm,
    right_margin = -6 * Plots.mm
)
k = @df sim_1_avg[sim_1_avg.encode_fidelity .== 0.9, :] plot(
    :focal_sim,
    :n_contenders_mean,
    groups = :probe_type_2,
    ylims = (0, 3),
    grid = false,
    label = false,
    showaxis = :x,
    title = "L = 0.9",
    left_margin = 0 * Plots.mm,
    right_margin = -6 * Plots.mm
)
l = @df sim_1_avg[sim_1_avg.encode_fidelity .== 1.0, :] plot(
    :focal_sim,
    :n_contenders_mean,
    groups = :probe_type_2,
    ylims = (0, 3),
    grid = false,
    label = false,
    showaxis = :x,
    title = "L = 1.0",
    left_margin = 0 * Plots.mm
)
plot(
    g,
    h,
    i,
    j,
    k,
    l,
    layout = lay,
    size = (1_200, 400),
    plot_title = "Number of Contenders",
    xlabel = "S"
)
