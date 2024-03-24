using StatsBase

function compute_similarity(trace::AbstractVector{<:Real}, probe::AbstractVector{<:Real})
    s = 0
    n = 0
    for i ∈ 1:length(probe)
        s += trace[i] * probe[i]
        n += is_not_null(trace[i], probe[i])
    end
    return s ./ n
end

function compute_similarity(traces::AbstractArray{<:Real,2}, probe::AbstractVector{<:Real})
    return [compute_similarity(r, probe) for r ∈ eachcol(traces)]
end

is_not_null(x1, x2) = (x1 ≠ 0) || (x2 ≠ 0)

compute_activations(traces, probe) = compute_similarity(traces, probe) .^ 3

function compute_cond_echo_content(
    activation::AbstractVector{<:Real},
    trace::AbstractVector{<:Real},
    τ::Real,
)
    x = 0.0
    for i ∈ 1:length(trace)
        x += activation[i] ≥ τ ? activation[i] * trace[i] : 0.0
    end
    return x
end

function compute_cond_echo_content(
    activations::AbstractVector{<:Real},
    traces::AbstractArray{<:Real,2},
    τ::Real,
)
    x = fill(0.0, size(traces, 1))
    i = 1
    for trace ∈ eachrow(traces)
        x[i] = compute_cond_echo_content(activations, trace, τ)
        i += 1
    end
    return x
end

function create_unspecified_probe(activations, traces, hypotheses, τ)
    probe = fill(0.0, size(traces, 1) + size(hypotheses, 1))
    i = 1
    for trace ∈ eachrow(traces)
        probe[i] = compute_cond_echo_content(activations, trace, τ)
        i += 1
    end
    for hypothesis ∈ eachrow(hypotheses)
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

function normalize(act)
    s = 0.0
    x = similar(act)
    for i ∈ 1:length(act)
        x[i] = act[i] ≥ 0 ? act[i] : 0.0
        s += act[i]
    end
    x ./= s
    return x
end

function compute_cond_echo_intensity(trace::AbstractVector{<:Real}, probe)
    return compute_activations(trace, probe)
end

function compute_cond_echo_intensity(
    activations,
    traces::AbstractArray{<:Real,2},
    probe::AbstractVector{<:Real},
    τ,
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

function compute_cond_echo_intensities(
    activations,
    traces::AbstractArray{<:Real,2},
    probes::AbstractArray{<:Real,2},
    τ,
)
    echo_intensities = fill(0.0, size(probes, 2))
    i = 1
    for probe ∈ eachcol(probes)
        echo_intensities[i] = compute_cond_echo_intensity(activations, traces, probe, τ)
        i += 1
    end
    return echo_intensities
end

function populate_working_memory!(model::AbstractHyGeneModel, semantic_activation)
    (; t_max, working_memory) = model
    empty!(working_memory)
    # retrieval failures
    n_fails = 0
    # retrieval probabilities (negative activation → zero probability)
    retrieval_probs = normalize(semantic_activation)
    n_hypotheses = length(retrieval_probs)
    # probability weights
    w = Weights(retrieval_probs)
    # dynamic threshold for SOC
    τₛ = 0.0
    while n_fails < t_max
        idx = sample(1:n_hypotheses, w)
        n_fails, τₛ = update_working_memory!(model, semantic_activation, idx, n_fails, τₛ)
    end
    sort!(working_memory)
    return nothing
end

function update_working_memory!(
    model::AbstractHyGeneModel,
    semantic_activation,
    idx,
    n_fails,
    τₛ,
)
    (; working_memory, κ, t_max) = model
    act = semantic_activation[idx]
    if (idx ∉ working_memory) && (act > τₛ)
        if length(working_memory) == κ
            _, min_idx = findmin(@view semantic_activation[working_memory])
            deleteat!(working_memory, min_idx)
        end
        push!(working_memory, idx)
        τₛ = minimum(@view semantic_activation[working_memory])
    else
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
        compute_cond_echo_intensities(activations, hypotheses, semantic_probes, τ)
    populate_working_memory!(model, semantic_activation)
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