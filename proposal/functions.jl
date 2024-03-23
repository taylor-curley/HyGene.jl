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
    threshold::Real,
)
    x = 0.0
    for i ∈ 1:length(trace)
        x += activation[i] ≥ threshold ? activation[i] * trace[i] : 0.0
    end
    return x
end

function compute_cond_echo_content(
    activations::AbstractVector{<:Real},
    traces::AbstractArray{<:Real,2},
    threshold::Real,
)
    x = fill(0.0, size(traces, 1))
    i = 1
    for trace ∈ eachrow(traces)
        x[i] = compute_cond_echo_content(activations, trace, threshold)
        i += 1
    end
    return x
end

function create_unspecified_probe(activations, traces, hypotheses, threshold)
    probe = fill(0.0, size(traces, 1) + size(hypotheses, 1))
    i = 1
    for trace ∈ eachrow(traces)
        probe[i] = compute_cond_echo_content(activations, trace, threshold)
        i += 1
    end
    for hypothesis ∈ eachrow(hypotheses)
        probe[i] = compute_cond_echo_content(activations, hypothesis, threshold)
        i += 1
    end
    max_act = maximum(probe)
    for j ∈ 1:length(probe)
        usp = probe[j]
        probe[j] /= max_act
    end
    return probe
end

function normalize!(act, threshold)
    s = 0.0
    for i ∈ 1:length(act)
        act[i] = act[i] ≥ threshold ? act[i] : 0.0
        s += act[i]
    end
    act ./= s
    return act
end

function compute_cond_echo_intensity(trace::AbstractVector{<:Real}, probe)
    return compute_activations(trace, probe)
end

function compute_cond_echo_intensity(
    activations,
    traces::AbstractArray{<:Real,2},
    probe::AbstractVector{<:Real},
    threshold,
)
    echo_intensity = 0.0
    i = 1
    n = 0
    for trace ∈ eachcol(traces)
        if activations[i] ≥ threshold
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
    threshold,
)
    echo_intensities = fill(0.0, size(probes, 2))
    i = 1
    for probe ∈ eachcol(probes)
        echo_intensities[i] =
            compute_cond_echo_intensity(activations, traces, probe, threshold)
        i += 1
    end
    return echo_intensities
end
