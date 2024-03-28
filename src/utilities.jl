get_indices(d::NamedTuple) = mapreduce(v -> v, vcat, values(d))

"""
    function make_traces(ns...)

Generates an array of traces with an arbitrary size. By default, rows correspond to features and columns 
correspond to traces. 

# Arguments

- `ns...`: dimensions of the trace array 
"""
function make_traces(ns...)
    return rand(-1:1, ns...)
end

"""
    encode(feature, ρ)

Encode a single feature from the environment correctly with probability `ρ` or encode the feature as zero 
with probability `1 - ρ`.

# Arguments

- `feature`: the value of a feature 
- `ρ`: the probability of encoding the feature correctly 
"""
function encode(feature, ρ)
    return rand() ≤ ρ ? feature : zero(feature)
end

"""
    replicate_trace(trace::AbstractVector{<:T}, ρ::Real) where {T <: Real}

Creates a degraded copy of a trace. The probability the original feature value is correctly copied is 
controlled by `ρ`.

- `trace::AbstractVector{<:T}`: a memory trace or environment event 
- `ρ`: the probability of encoding the feature correctly 
"""
function replicate_trace(trace::AbstractVector{<:T}, ρ::Real) where {T <: Real}
    new_trace = copy(trace)
    for i ∈ 1:length(new_trace)
        new_trace[i] = encode(new_trace[i], ρ)
    end
    return new_trace
end

"""
    replicate_traces(
        traces::AbstractArray{<:T, 2},
        n_reps::AbstractArray{<:Real},
        ρ::Real
    ) where {T <: Real}

Creates a degraded copies of a matrix of traces. The probability the original feature value is correctly copied is 
controlled by `ρ`. 

- `traces::AbstractArray{<:T, 2}`: a memory trace or environment event 
- `n_reps::AbstractVector{<:Real}`: the number of degraded replications of traces in `traces`. The ith element in `n_reps`
    corresponds to the number of replicates of the trace defined in the ith column of `traces`. 
- `ρ::Real`: the probability of encoding the feature correctly 

# Returns 

- `new_traces::AbstractArray{<:Real}`: assuming `traces` is a 10 × 4 array and `n_reps = [1,1,2,2]`, the return array will be 
a 10 × 6 array. 

"""
function replicate_traces(
    traces::AbstractArray{<:T, 2},
    n_reps::AbstractVector{<:Real},
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
