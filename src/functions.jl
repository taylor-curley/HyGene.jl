"""
    compute_similarity(trace::AbstractVector{<:Real}, probe::AbstractVector{<:Real})

# Arguments 

- `trace::AbstractVector{<:Real}`: a vector representing a single memory trace 
- `probe::AbstractVector{<:Real}`: a vector representing a memory cue
"""
function compute_similarity(trace::AbstractVector{<:Real}, probe::AbstractVector{<:Real})
    s = 0
    n = 0
    for i ∈ 1:length(probe)
        s += trace[i] * probe[i]
        n += is_not_null(trace[i], probe[i])
    end
    return s ./ n
end

"""
    compute_similarity(trace::AbstractVector{<:Real}, probe::AbstractVector{<:Real})

# Arguments 

- `trace::AbstractVector{<:Real}`: a vector representing a single memory trace 
- `probe::AbstractVector{<:Real}`: a vector representing a memory cue
"""
function compute_similarity(traces::AbstractArray{<:Real, 2}, probe::AbstractVector{<:Real})
    return [compute_similarity(r, probe) for r ∈ eachcol(traces)]
end

is_not_null(x1, x2) = (x1 ≠ 0) || (x2 ≠ 0)

"""
    compute_similarity(trace::AbstractVector{<:Real}, probe::AbstractVector{<:Real})

# Arguments 

- `trace::AbstractVector{<:Real}`: a vector representing a single memory trace 
- `probe::AbstractVector{<:Real}`: a vector representing a memory cue
"""
compute_activations(traces, probe) = compute_similarity(traces, probe) .^ 3

"""
    compute_similarity(trace::AbstractVector{<:Real}, probe::AbstractVector{<:Real})

# Arguments 

- `trace::AbstractVector{<:Real}`: a vector representing a single memory trace 
- `probe::AbstractVector{<:Real}`: a vector representing a memory cue
"""
function compute_cond_echo_content(
    activation::AbstractVector{<:Real},
    trace::AbstractVector{<:Real},
    τ::Real
)
    x = 0.0
    for i ∈ 1:length(trace)
        x += activation[i] ≥ τ ? activation[i] * trace[i] : 0.0
    end
    return x
end

"""
    compute_similarity(trace::AbstractVector{<:Real}, probe::AbstractVector{<:Real})

# Arguments 

- `trace::AbstractVector{<:Real}`: a vector representing a single memory trace 
- `probe::AbstractVector{<:Real}`: a vector representing a memory cue
"""
function compute_cond_echo_content(
    activations::AbstractVector{<:Real},
    traces::AbstractArray{<:Real, 2},
    τ::Real
)
    x = fill(0.0, size(traces, 1))
    i = 1
    for trace ∈ eachrow(traces)
        x[i] = compute_cond_echo_content(activations, trace, τ)
        i += 1
    end
    return x
end

"""
    create_unspecified_probe(
        activations::AbstractVector{<:Real},
        data_component::AbstractArray{<:Real, 2},
        hypothesis_component::AbstractArray{<:Real, 2},
        τ
    )

Creates an unspecified probe which is used to subsequently to compute activation of semantic memory traces. The unspecified probe consists of 
of a data and hypothesis component respectively. 

# Arguments 

- `activations::AbstractVector{<:Real}`: a vector of activations which typically represents the data component of episodic memory traces
- `traces::AbstractArray{<:Real, 2}`: the data component of episodic memory in which rows represent features and columns represent traces
- `traces::AbstractArray{<:Real, 2}`: the hypothesis component of episodic memory in which rows represent features and columns represent traces
- `probe::AbstractVector{<:Real}`: typically a the hypothesis component of a semantic trace, which is used to probe the hypothesis components of traces in episodic 
    memory
- `τ`: the activation threshold
"""
function create_unspecified_probe(
    activations::AbstractVector{<:Real},
    data_component::AbstractArray{<:Real, 2},
    hypothesis_component::AbstractArray{<:Real, 2},
    τ
)
    probe = fill(0.0, size(data_component, 1) + size(hypothesis_component, 1))
    i = 1
    for trace ∈ eachrow(data_component)
        probe[i] = compute_cond_echo_content(activations, trace, τ)
        i += 1
    end
    for hypothesis ∈ eachrow(hypothesis_component)
        probe[i] = compute_cond_echo_content(activations, hypothesis, τ)
        i += 1
    end
    max_act = maximum(probe)
    for j ∈ 1:length(probe)
        usp = probe[j]
        probe[j] /= max_act
    end
    return probe
end

"""
    normalize(activations::AbstractVector{<:Real})

Computes retrieval probabilities from semantic activations. Activation values below zero are set to zero
to prevent the corresponding semantic traces from being retrieved and added to working memory.

# Arguments 

- `activations::AbstractVector{<:Real}`: a vector of activations for semantic traces
"""
function normalize(act::AbstractVector{<:Real})
    s = 0.0
    x = similar(act)
    for i ∈ 1:length(act)
        x[i] = act[i] ≥ 0 ? act[i] : 0.0
        s += act[i]
    end
    x ./= s
    return x
end

"""
    compute_cond_echo_intensity(trace::AbstractVector{<:Real}, probe::AbstractVector{<:Real})

Computes the conditional echo intensity of a given memory trace. 

# Arguments 

- `trace::AbstractVector{<:Real}`: a vector representing a single memory trace 
- `probe::AbstractVector{<:Real}`: a vector representing a memory cue
"""
function compute_cond_echo_intensity(
    trace::AbstractVector{<:Real},
    probe::AbstractVector{<:Real}
)
    return compute_activations(trace, probe)
end

"""
    compute_cond_echo_intensity(
        activations::AbstractVector{<:Real},
        traces::AbstractArray{<:Real,2},
        probe::AbstractVector{<:Real},
        τ,
    )

Computes the conditional echo intensity. Typically, this is computed on the hypothesis component of episodic memory traces using 
    the hypothesis component of the semantic memory trace as a memory probe.    

# Arguments 

- `activations::AbstractVector{<:Real}`: a vector of activations which typically represents the data component of episodic memory traces
- `traces::AbstractArray{<:Real,2}`: a memory store in which rows represent features and columns represent traces. Typically, this store represents 
    the hypothesis component of episodic memory traces.
- `probe::AbstractVector{<:Real}`: typically a the hypothesis component of a semantic trace, which is used to probe the hypothesis components of traces in episodic 
    memory
- `τ`: the activation threshold
"""
function compute_cond_echo_intensity(
    activations::AbstractVector{<:Real},
    traces::AbstractArray{<:Real, 2},
    probe::AbstractVector{<:Real},
    τ
)
    echo_intensity = 0.0
    i = 1
    n = 0
    for trace ∈ eachcol(traces)
        if activations[i] ≥ τ
            echo_intensity += compute_cond_echo_intensity(trace, probe)
            n += 1
        end
        i += 1
    end
    return echo_intensity / n
end

"""
    compute_cond_echo_intensities(
        activations::AbstractVector{<:Real},
        traces::AbstractArray{<:Real,2},
        probes::AbstractArray{<:Real,2},
        τ,
    )

Computes the conditional echo intensities for multiple probes. Typically, this is computed on the hypothesis component of episodic memory traces using 
    the hypothesis component of the semantic memory trace as a memory probe. 

# Arguments 

- `activations::AbstractVector{<:Real}`: a vector of activations which typically represents the data component of episodic memory traces
- `traces::AbstractArray{<:Real,2}`: a memory store in which rows represent features and columns represent traces. Typically, this store represents 
    the hypothesis component of episodic memory traces.
- `probes::AbstractArray{<:Real,2},`: a set of probes used to compute conditional echo intensity. Each column corresponds to a memory probe. 
    Typically, the probes are the hypothesis components of semantic traces
- `τ`: the activation threshold
"""
function compute_cond_echo_intensities(
    activations::AbstractVector{<:Real},
    traces::AbstractArray{<:Real, 2},
    probes::AbstractArray{<:Real, 2},
    τ
)
    echo_intensities = fill(0.0, size(probes, 2))
    i = 1
    for probe ∈ eachcol(probes)
        echo_intensities[i] = compute_cond_echo_intensity(activations, traces, probe, τ)
        i += 1
    end
    return echo_intensities
end

"""
    populate_working_memory!(model::AbstractHyGeneModel, semantic_activations::AbstractVector{<:Real})

Adds indices corresponding to semantic traces to a vector representing working memory. The working memory vector 
is cleared before semantic traces are added. 

# Arguments 

- `model::AbstractHyGeneModel`: an HyGene model which is a subtype of `AbstractHyGeneModel`
- `probe::AbstractVector{<:Real}`: a vector representing a memory cue
"""
function populate_working_memory!(
    model::AbstractHyGeneModel,
    semantic_activations::AbstractVector{<:Real}
)
    (; t_max, working_memory) = model
    empty!(working_memory)
    # retrieval failures
    n_fails = 0
    # retrieval probabilities (negative activation → zero probability)
    retrieval_probs = normalize(semantic_activations)
    n_hypotheses = length(retrieval_probs)
    # probability weights
    w = Weights(retrieval_probs)
    # dynamic threshold for SOC
    τₛ = 0.0
    while n_fails < t_max
        idx = sample(1:n_hypotheses, w)
        n_fails, τₛ = update_working_memory!(model, semantic_activations, idx, n_fails, τₛ)
    end
    sort!(working_memory)
    return nothing
end

function update_working_memory!(
    model::AbstractHyGeneModel,
    semantic_activations,
    idx,
    n_fails,
    τₛ
)
    (; working_memory, κ, t_max) = model
    act = semantic_activations[idx]

    if (idx ∉ working_memory) && (act > τₛ)
        if length(working_memory) == κ
            _, min_idx = findmin(semantic_activations[working_memory])
            deleteat!(working_memory, min_idx)
        end
        n_fails = 0
        push!(working_memory, idx)
        τₛ = minimum(semantic_activations[working_memory])
    else
        # if already in working memory, or below threshold, then failure
        n_fails += 1
    end
    return n_fails, τₛ
end

function judge_hypotheses(model::AbstractHyGeneModel, probe)
    (; semantic_memory, τ) = model
    d_idx = get_indices(model.data_map)
    h_idx = get_indices(model.hypothesis_map)
    data_traces = @view model.episodic_memory[d_idx, :]
    semantic_probes = @view model.semantic_memory[h_idx, :]
    hypothesis_traces = @view model.episodic_memory[h_idx, :]
    activations = compute_activations(data_traces, probe)
    unspecified_probe =
        create_unspecified_probe(activations, data_traces, hypothesis_traces, τ)
    semantic_activation = compute_activations(semantic_memory, unspecified_probe)
    retrieval_probs = normalize(semantic_activation)
    echo_intensities =
        compute_cond_echo_intensities(activations, hypothesis_traces, semantic_probes, τ)
    populate_working_memory!(model, echo_intensities)
    return judge_posterior(model, echo_intensities)
end

function judge_hypotheses(model::AbstractHyGeneModel, probe, data_components::NamedTuple)
    # 1. compute activation
    # 2. compute unspecified_probe
    # 3. compute retrieval_probs
    # 4. populate working memory 
    # 5. judge posterior probabilities

end

function judge_posterior(model::AbstractHyGeneModel, echo_intensities)
    ei = echo_intensities[model.working_memory]
    return ei ./ sum(ei)
end
