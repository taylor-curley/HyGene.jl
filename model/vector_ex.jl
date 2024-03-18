# Prelims
cd(@__DIR__)
using Pkg
# use package environment
Pkg.activate("")
using StatsBase, StatsPlots, ProgressMeter, DataFrames, Plots

create_vec(n_values::Integer) = rand([-1.0, 0.0, 1.0], n_values)

function trace_replication(trace::Vector{<:Number}, similarity::Number, decay::Bool = false)
    out = zeros(length(trace))
    for i in eachindex(trace)
        if trace[i] != 0.0
            if rand() > similarity
                out[i] = decay ? 0.0 : rand([-1.0, 0.0, 1.0])
            else
                out[i] = deepcopy(trace[i])
            end
        end
    end
    return out
end

function trace_similarity(probe::Vector{<:Number}, trace::Vector{<:Number})
    length(probe) == length(trace) || throw(
        DimensionMismatch(
            "the lengths of the probe (n=$(length(probe))) and trace (n=$(length(trace))) vectors must be the same!",
        ),
    )
    N = length(probe)
    for (p, t) in zip(probe, trace)
        (p .== 0.0) && (t .== 0.0) ? N -= 1 : nothing
    end
    return (probe'trace) / float(N)
end

function trace_activation(probe::Vector{<:Number}, trace::Vector{<:Number})
    return trace_similarity(probe, trace)^3
end

# Parameters
n_features = 15
n_minivecs = 10
n_hypotheses = 10
t_max = 10
A_c = 0.217
focal_sim = [0.0, 0.25, 0.45, 0.65, 0.85, 1.0]
encode_fidelity = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
d_obs_sim = 0.85
trace_dists = Vector{Integer}(vcat([70], ones(n_hypotheses - 1) * 10))
n_trials = 10
n_participants = 100

# Create simulator function
function sim_controller(
    n_features,
    n_minivecs,
    n_hypotheses,
    t_max,
    A_c,
    S,
    L,
    d_obs_sim,
    trace_dists,
    n_trials,
)
    # Create focal hypothesis prototype
    focal_prototype = create_vec(n_features * n_minivecs)

    # Replicate focal prototype to create alternative prototypes
    # There should be some similiarity between focal and alt prototypes
    alt_prototypes = [trace_replication(focal_prototype, S) for _ = 1:(n_hypotheses-1)]

    # Join prototypes
    prototypes = vcat([focal_prototype], alt_prototypes)

    # Create long-term memory with set number of traces for each prototype
    # Similarity between prototype and trace set by encoding parameter
    # Also create semantic memory traces by getting averages of representations
    long_term_memory = []
    semantic_memory = []
    for (proto, reps) in zip(prototypes, trace_dists)
        sem_trace = zeros(n_minivecs * n_features)
        for _ = 1:reps
            new_trace = trace_replication(proto, L)
            sem_trace .+= new_trace
            push!(long_term_memory, new_trace)
        end
        push!(semantic_memory, sem_trace / reps)
    end

    # Run through trials
    results = zeros(n_trials, 6)
    for n = 1:n_trials
        # Create probe from focal prototype
        probe = trace_replication(focal_prototype, d_obs_sim)

        # Generate conditional echo content
        echo = zeros(n_minivecs * n_features)
        for item in long_term_memory
            A_i = trace_activation(probe, item)
            # If activation is greater than threshold, multiply contents by
            # activation and add to echo content
            if A_i > A_c
                cond_trace = copy(item) .* A_i
                echo += cond_trace
            end
        end

        # Standardize echo content vector
        echo ./= maximum(abs.(echo))

        # Fill set of hypotheses by bouncing echo off of semantic memory
        soc = []
        soc_acts = []
        soc_pos_in_sem = []
        t = 0
        act_min = 0.0
        act_weights = [trace_activation(echo, trace) for trace in semantic_memory]
        act_weights[act_weights.<0.0] .= 0.0
        act_weights = map(x -> isnan(x) ? zero(x) : x, act_weights)
        act_weights ./ sum(act_weights)
        while t < t_max
            sem_trace = copy(sample(semantic_memory, Weights(act_weights)))
            A_c = trace_activation(echo, sem_trace)
            if A_c > act_min
                push!(soc, sem_trace)
                push!(soc_acts, A_c)
                push!(soc_pos_in_sem, argmax([sem_trace == sem for sem in semantic_memory]))
                act_min = copy(A_c)
            else
                t += 1
            end
        end
        # Find highest-activated hypothesis
        if (length(soc) > 0)
            winner = soc_pos_in_sem[argmax(soc_acts)]
        else
            winner = 0
        end

        # Push to vector
        results[n, :] .= [S, L, n, length(soc), winner, (winner == 1) * 1]
    end
    columns = ["S", "L", "Trial", "N_SOC", "Winner", "ACC"]
    return DataFrame(results, columns)
end

# Loop through participants
# Instantiate blank data frame
results = DataFrame(
    S = Float64[],
    L = Float64[],
    Trial = Number[],
    N_SOC = Number[],
    Winner = Number[],
    ACC = Number[],
)

for _ = 1:n_participants
    for S in focal_sim
        for L in encode_fidelity
            out = sim_controller(
                n_features,
                n_minivecs,
                n_hypotheses,
                t_max,
                A_c,
                S,
                L,
                d_obs_sim,
                trace_dists,
                n_trials,
            )
            results = vcat(results, out)
        end
    end
end

sim_avgs = combine(groupby(results, [:S, :L]), [:N_SOC, :ACC] .=> mean)

lay = @layout [a b c d e f]
a = @df sim_avgs[sim_avgs.L.==0.1, :] plot(
    :S,
    :ACC_mean,
    lims = (0, 1),
    grid = false,
    title = "L = 0.1",
)
b = @df sim_avgs[sim_avgs.L.==0.3, :] plot(
    :S,
    :ACC_mean,
    lims = (0, 1),
    grid = false,
    title = "L = 0.3",
)
c = @df sim_avgs[sim_avgs.L.==0.5, :] plot(
    :S,
    :ACC_mean,
    lims = (0, 1),
    grid = false,
    title = "L = 0.5",
)
d = @df sim_avgs[sim_avgs.L.==0.7, :] plot(
    :S,
    :ACC_mean,
    lims = (0, 1),
    grid = false,
    title = "L = 0.7",
)
e = @df sim_avgs[sim_avgs.L.==0.9, :] plot(
    :S,
    :ACC_mean,
    lims = (0, 1),
    grid = false,
    title = "L = 0.9",
)
f = @df sim_avgs[sim_avgs.L.==1.0, :] plot(
    :S,
    :ACC_mean,
    lims = (0, 1),
    grid = false,
    title = "L = 1.0",
)
plot(
    a,
    b,
    c,
    d,
    e,
    f,
    layout = lay,
    size = (1_200, 300),
    plot_title = "Average Accuracy",
    xlabel = "S",
)

g = @df sim_avgs[sim_avgs.L.==0.1, :] plot(
    :S,
    :N_SOC_mean,
    ylims = (0, 3),
    grid = false,
    title = "L = 0.1",
)
h = @df sim_avgs[sim_avgs.L.==0.3, :] plot(
    :S,
    :N_SOC_mean,
    ylims = (0, 3),
    grid = false,
    title = "L = 0.3",
)
i = @df sim_avgs[sim_avgs.L.==0.5, :] plot(
    :S,
    :N_SOC_mean,
    ylims = (0, 3),
    grid = false,
    title = "L = 0.5",
)
j = @df sim_avgs[sim_avgs.L.==0.7, :] plot(
    :S,
    :N_SOC_mean,
    ylims = (0, 3),
    grid = false,
    title = "L = 0.7",
)
k = @df sim_avgs[sim_avgs.L.==0.9, :] plot(
    :S,
    :N_SOC_mean,
    ylims = (0, 3),
    grid = false,
    title = "L = 0.9",
)
l = @df sim_avgs[sim_avgs.L.==1.0, :] plot(
    :S,
    :N_SOC_mean,
    ylims = (0, 3),
    grid = false,
    title = "L = 1.0",
)
plot(
    g,
    h,
    i,
    j,
    k,
    l,
    layout = lay,
    size = (1_200, 300),
    plot_title = "Number of Contenders",
    xlabel = "S",
)
