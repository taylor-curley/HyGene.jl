get_indices(d::NamedTuple) = mapreduce(v -> v, vcat, values(d))

function make_traces(ns...)
    return rand(-1:1, ns...)
end

function encode(feature, ρ)
    return rand() ≤ ρ ? feature : zero(feature)
end

function replicate_trace(trace::AbstractVector{<:T}, ρ::Real) where {T <: Real}
    new_trace = copy(trace)
    for i ∈ 1:length(new_trace)
        new_trace[i] = encode(new_trace[i], ρ)
    end
    return new_trace
end

function replicate_traces(
    traces::AbstractArray{<:T, 2},
    n_reps::AbstractArray{<:Real},
    ρ::Real
) where {T <: Real}
    n_total = sum(n_reps)
    n_features = size(traces, 1)
    new_traces = zeros(T, n_features, n_total)
    i, r = 1, 1
    for trace ∈ eachcol(traces)
        for _ ∈ 1:n_reps[r]
            new_traces[:, i] = replicate_trace(trace, ρ)
            i += 1
        end
        r += 1
    end
    return new_traces
end
