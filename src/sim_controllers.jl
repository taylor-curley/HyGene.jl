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
  - `n_items`: Number of items to generate. Note that using `generate_item` to create lists will create items with no similarity to each other.

# Returns

  - A single vector or collection of minivectors.

"""
generate_item(n_values::Integer) = Vector{Float64}(sample([-1.0,0.0,1.0],n_values))
function generate_item(n_values::Integer, n_contexts::Integer)
    return [generate_item(n_values) for _ in 1:n_contexts]
end
function generate_item(n_values::Integer, n_contexts::Integer, n_items::Integer)
    return [generate_item(n_values, n_contexts) for _ in 1:n_items]
end

"""
    generate_observation(label, contexts::Vector{<:Context}, hypothesis::Hypothesis)

High-level constructor of `Observaton` objects given 1 or more contexts and a single
hypothesis. 

# Arguments

  - `label`: Generic label for the observation.
  - `contexts`: One or more objects of the `Context` type.
  - `hypothesis`: A single object of type `Hypothesis`.

# Returns

  - Object of type `Observation`.

"""
function generate_observation(label, contexts::Vector{<:Context}, hypothesis::Hypothesis)
    content = Vector{Vector{Float64}}(undef,0)
    for c in contexts
        push!(content,c.content)
    end
    push!(content, hypothesis.content)
    return Observation(label, length(hypothesis.content), length(contexts), content)
end


"""
    trace_decay(vector::Vector{<:Real}, decay<:AbstractFloat)
    trace_decay(vector::Vector{Vector{<:Real}}, decay::AbstractFloat)

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
function trace_decay(vector::Vector{<:Vector{<:Real}}, decay::AbstractFloat)
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
    trace_similarity(probe::Vector{<:Real}, trace::Vector{<:Real})
    trace_similarity(probe::Vector{<:Real}, trace::Vector{<:Vector{<:Real}})
    trace_similarity(probe::Vector{<:Vector{<:Real}}, trace::Vector{<:Vector{<:Real}})
    trace_similarity(probe::Vector{<:Vector{<:Real}}, trace::HypothesisGeneration)
    trace_similarity(probe::HypothesisGeneration, trace::Vector{<:HypothesisGeneration})
    trace_similarity(probe::Vector{<:Real}, trace::Vector{<:HypothesisGeneration})
    trace_similarity(probe::Vector{<:Vector{<:Real}}, trace::Vector{<:HypothesisGeneration})
    trace_similarity(probe::HypothesisGeneration, trace::HypothesisGeneration)

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
function trace_similarity(probe::Vector{<:Real},trace::Vector{<:Vector{<:Real}})
    out = []
    for t in trace
        push!(out, trace_similarity(probe,t))
    end
    return Vector{Float64}(out)
end
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
function trace_similarity(probe::Vector{<:Real},trace::Vector{<:HypothesisGeneration})
    out = []
    for item in trace
        push!(out, trace_similarity(probe,item.content))
    end
    return out
end
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
    obs_to_trace(obs::Observation, decay::Real)

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


"""
    echo_content(act::Real, trace::Vector{<:Real})
    echo_content(probe::HypothesisGeneration, store::Vector{<:HypothesisGeneration})

Returns the sum of activation values multiplied against trace item vectors. If the input is a memory store,
then the values are normalized within each mini-vector.

# Arguments

  - `act`: Activation value to multiply against a given trace vector.
  - `trace`: Trace vector from a memory store.
  - `probe`: Input object or vector to compare against memory
  - `store`: A memory store; usually `HygeneModel.long_term_memory` or `HyGeneModel.semantic_memory`

# Returns

 - Vector or collection of mini-vectors containing echo content.

"""
echo_content(act::Real, trace::Vector{<:Real}) = Vector{Float64}(act .* trace)
function echo_content(act::Real, trace::Vector{<:Vector{<:Real}})
    out = []
    for t in trace
        push!(out, echo_content(act,t))
    end
    return Vector{Vector{Float64}}(out)
end
function echo_content(probe::HypothesisGeneration, store::Vector{<:HypothesisGeneration})
    # Quick input type check
    (typeof(probe) <: HyGeneModel) ? error("Incorrect specification.") : nothing
    # Get trace activations
    acts = trace_activation(probe, store)
    # Begin echo content collection
    out = echo_content(acts[1],store[1].content)
    for (act,trace) in zip(acts[2:end],store[2:end])
        out .+= echo_content(act,trace.content)
    end
    # Normalize within mini-vectors
    # Make sure values are within [-1.0,1.0]
    for i in 1:length(out)
        out[i] ./= maximum(abs.(out[i]))
        out[i][out[i].>1.0] .= 1.0
        out[i][out[i].<-1.0] .= -1.0
    end
    # Return values
    if (typeof(probe) <: Trace) | (typeof(probe) <: Observation)
        return Vector{Vector{Float64}}(out)
    else
        return Vector{Float64}(out)
    end
end


"""
    get_contenders!(probe::HypothesisGeneration, model::HyGeneModel)

Populates the Set of Contenders (SoC; or `HyGeneModel.working_memory`) with candidate
traces greater than the adaptive activation threshold. The procedure ends when the
number of retrieval failures is greater than `t_max`.

# Arguments

  - `probe`: Information to compare against memory.
  - `model`: Overall `HyGeneModel` object with items in long-term or semantic memory.

"""
function get_contenders!(probe::HypothesisGeneration, model::HyGeneModel)
    model.working_memory = Vector{HypothesisGeneration}(undef,0)                    # Flush working memory store (SoC)
    model.act_min_h = model.act_min_reset                                           # Reset floating minimum activation threshold
    model.t = 0                                                                     # Reset retrieval failure count
    while model.t < model.t_max                                                     # End procedure at max number of retrieval failures
        trace = sample(1:length(model.long_term_memory))                            # Sample random item from memory store
        trace_act = trace_activation(probe.content, model.long_term_memory[trace])  # Compare similarity of probe and memory trace
        if trace_act > model.act_min_h                                              # If the activation of the trace is greater than the floating threshold...
            push!(model.working_memory, model.long_term_memory[trace])              # copy trace to working memory store...
            model.act_min_h = trace_act                                             # and set minimum activation value to newly-observed activation.
        else                                                                        # Otherwise...
            model.t += 1                                                            # increase the retrieval failure count.
        end
    end
end