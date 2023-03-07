"""
    generate_item(n_values::Integer)
    generate_item(n_values:Integer, n_contexts:Integer)

General function for creating minivectors and collections of minivectors. Multiple
dispatch methods allow the function to return either one minivector of length `n_values`
or a collection of minivectors of length `n_contexts`, each of length `n_values`.
For now, selection of 0, -1, and 1 are selected with equal probability (33%).

# Arguments

  - `n_values`: Length of feature mini-vectors.
  - `n_contexts`: Number of mini-vectors within a single item.

# Returns

  - A single vector or collection of minivectors.

"""
generate_item(n_values::Integer) = Vector{Float64}(sample([-1.0,0.0,1.0],n_values))
function generate_item(n_values::Integer, n_contexts::Integer)
    return [generate_item(n_values) for _ in 1:n_contexts]
end



"""
    trace_decay(vector::Vector{<:Real}, decay<:AbstractFloat)

General function that changes each number within a vector by probability `decay`.
Mostly used in conjunction with the `trace_decay!` append function.

# Arguments

  - `vector`: A vector of float values that are either `0.0`, `1.0`, or `-1.0`.
  - `decay`: A float value indicating the individual probability that each number will change to `0.0`.

# Returns

  - Vector of float or integer values that have been changed to `0.0` with probability `decay`.

"""
function trace_decay(vector::Vector{<:Real}, decay::AbstractFloat)
    out = deepcopy(vector)
    for i in 1:length(vector)
        rand() < decay ? out[i] = 0.0 : nothing
    end
    return out
end
function trace_decay(vector::Vector{Vector{<:Real}}, decay::AbstractFloat)
    out = Vector{<:Real}[]
    for minivec in vector
        push!(out, trace_decay(minivec,decay))
    end
    return out
end


"""
    trace_decay!(trace::Trace, decay<:AbstractFloat)
    trace_decay!(model::HyGeneModel)

Function that changes each number within a vector by probability `decay`. The 
updated vectors are written back to their respective `content` fields. If there
are multiple minivectors, then it will loop through the collection.

# Arguments

  - `trace`: Object of type `Trace`.
  - `model`: Object of type `HyGeneModel`.
  - `decay`: A float value indicating the individual probability that each number will change to `0.0`.

"""
function trace_decay!(trace::Trace, decay::AbstractFloat)
    # Check if there are multiple minivectors
    # If so, loop through them; else, loop through the single vector.
    if typeof(trace.content) <: Vector{Vector{Float64}}
        for i in 1:length(trace.content)
            trace.content[i] = trace_decay(trace.content[i], decay)
        end
    else 
        trace.content = trace_decay(trace.content, decay)
    end
end
function trace_decay!(model::HyGeneModel)
    for item in model.long_term_memory
        trace_decay!(item, model.decay)
    end
end

"""
    trace_similarity(probe::Vector{<:Real},trace::Vector{<:Real})
    trace_similarity(probe::Vector{<:Vector{<:Real}},trace::Vector{<:Vector{<:Real}})
    trace_similarity(probe::Vector{<:Vector{<:Real}},trace::HypothesisGeneration)
    trace_similarity(probe::Vector{<:Vector{<:Real}},trace::Vector{<:HypothesisGeneration})
    trace_similarity(probe::HypothesisGeneration,trace::HypothesisGeneration)

Calculates the similarity between a probe item and a trace. Eq 1 in Thomas et al. (2008).
Several potential inputs have been proposed using Julia's multiple dispatch capabilities.
You may need to include more as the model evolves.

# Arguments

  - `probe`: An input vector or collection of minivectors.
  - `trace`: A vector or collection of minivectors to be compared against the probe.

# Returns

  - Float value(s) denoting similarity between probe and trace. 

"""
trace_similarity(probe::Vector{<:Real},trace::Vector{<:Real}) = (probe'trace) / sum((abs.(probe) .+ abs.(trace)).!=0.0)
function trace_similarity(probe::Vector{<:Vector{<:Real}},trace::Vector{<:Vector{<:Real}})
    s = 0.0
    n = 0.0
    for i in 1:length(probe)
        s += probe[i]'trace[i]                              # Inner product
        n += sum((abs.(probe[i]) .+ abs.(trace[i])).!=0.0)  # Number of non-zero pairs
    end
    return s/n
end
trace_similarity(probe::Vector{<:Vector{<:Real}},trace::HypothesisGeneration) = trace_similarity(probe,trace.content)
trace_similarity(probe::HypothesisGeneration,trace::Vector{<:HypothesisGeneration}) = trace_similarity(probe.content,trace)
function trace_similarity(probe::Vector{<:Vector{<:Real}},trace::Vector{<:HypothesisGeneration})
    out = []
    for item in trace
        push!(out, trace_similarity(probe,item.content))
    end
    return out
end
trace_similarity(probe::HypothesisGeneration,trace::HypothesisGeneration) = trace_similarity(probe.content,trace.content)



"""
    trace_activation(probe,trace,power=3.0)

The positively accelerated function of `trace_similarity()` to the power specified by the user.
Eq 2 in Thomas et al. (2008). Input type definitions are not provided here since the main 
interface is `trace_similarity()`, which is multiply defined.

# Arguments

  - `probe`: An input vector or collection of minivectors.
  - `trace`: A vector or collection of minivectors to be compared against the probe.
  - `power`: Exponent of the activation acceleration function. Usually set to `3.0`.

# Returns

  - Float value(s) denoting accelerated activation between a probe and trace.

"""
function trace_activation(probe,trace,power=3.0) 
    if length(trace_similarity(probe,trace)) == 1
        return trace_similarity(probe,trace)^power
    else
        return trace_similarity(probe,trace).^power
    end
end

"""
    echo_intensity(probe,trace)

The sum of trace activation values across a field of traces given a probe.
Eq. 3 in Thomas et al. (2008).

# Arguments

  - `probe`: An input vector or collection of minivectors.
  - `trace`: A vector or collection of minivectors to be compared against the probe.
  - `power`: Exponent of the activation acceleration function. Usually set to `3.0`.

# Returns

  - Float value denoting echo intensity.

"""
echo_intensity(probe,trace) = sum(trace_activation(probe,trace))


"""
    conditional_echo_intensity(probe,trace,threshold)

Returns the conditional echo intensity for a set of items that exceed a given activation
threshold. Eq. 5 in Thomas et al. (2008).

# Arguments

  - `probe`: An input vector or collection of minivectors.
  - `trace`: A vector or collection of minivectors to be compared against the probe.

"""
function conditional_echo_intensity(probe,trace,threshold)
    acts = trace_activation(probe,trace)
    valid_acts = findall(x -> x > threshold, acts)
    if isempty(valid_acts)
        return 0.0
    elseif length(valid_acts) == 1
        return acts[valid_acts]^3
    else
        return echo_intensity(probe,trace[valid_acts]) / length(valid_acts)
    end
    return 
end
conditional_echo_intensity(probe::Observation,trace::HyGeneModel) = conditional_echo_intensity(probe.content,trace.long_term_memory,trace.act_thresh)

"""
    obs_to_trace(obs::Observation, decay)

Converts observations to traces by reading the content of an `Observation` object, degrading
the vector contents with probability `decay`, and creating a new `Trace` object with the 
altered vector. Mostly to assist formation of `long_term_memory` in the `HyGeneModel` object.

# Arguments

  - `obs`: An object of type `Observation`.
  - `decay`: Float value between `0` and `1` indicating the degree to which an item should decay. 

# Returns

  - Object of type `Trace` with an item vector degraded with probability `decay`.

"""
function obs_to_trace(obs::Observation, decay::Real)
    vals = (obs.label,obs.n_values,obs.n_contexts,0)
    trace = trace_decay(obs.content, decay)
    return Trace(vals..., trace)
end


"""
    trace_replicator(trace::Vector{<:Real}, similarity::Real)
    trace_replicator(trace::HypothesisGeneration, similarity::Real)

Copies an item vector with probability `similarity` to a new object. Non-similar features are
randomly given a new value, i.e. `[-1,0,1]`.

# Arguments

  - `trace`: Either an item vector or simulation object containing an item vector.
  - `similarity`: Degree of similarity between original and new item vectors.

# Returns

  - `out`: Replicated object with an imperfect copy of the original item vector.

"""
function trace_replicator(trace::Vector{<:Real}, similarity::Real)
    out = deepcopy(trace)
    for i in 1:length(out)
        rand() < similarity ? (out[i] = sample([-1.0,0.0,1.0])) : nothing
    end
    return out
end
function trace_replicator(trace::HypothesisGeneration, similarity::Real)
    out = deepcopy(trace)
    out.content = trace_replicator(trace.content, similarity)
    return out
end