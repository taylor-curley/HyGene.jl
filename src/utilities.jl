"""
    create_vec(n_values::Integer)

Returns a vector of length `n_values` with numbers randomly chosen from `[-1.0 0.0 1.0]`.

# Arguments

  - `n_values::Integer`: Number of features in resulting vector.

# Example

```jldoctest
julia> using Random; Random.seed!(1);

julia> create_vec(5)
5-element Vector{Float64}:
 -1.0
  0.0
  1.0
  0.0
  1.0
```
"""
create_vec(n_values::Integer) = rand(-1.0:1, n_values)


"""
    create_vec(n_vecs::Integer, n_values::Integer)

Returns an array of minivectors. The array is length `n_vecs` while each minivector has `n_values` 
features randomly sampled from `[-1.0 0.0 1.0]`.

# Arguments

  - `n_vecs::Integer`: Number of minivectors.
  - `n_values::Integer`: Number of features in each mini-vector.

# Example

```jldoctest
julia> using Random; Random.seed!(1);

julia> create_vec(2, 3)
2-element Vector{Vector{Float64}}:
 [0.0, -1.0, -1.0]
 [0.0, 0.0, -1.0]
```
"""
create_vec(n_vecs::Integer, n_values::Integer) = [create_vec(n_values) for _ in 1:n_vecs]


"""
    create_vec(n_vecs::Integer, n_values::Integer, similarity::Number)

Returns an array of length `n_vecs` with each minivector containing `n_values` features
randomly sampled from `[-1.0 0.0 1.0]`. After an initial minivector is created, the
subsequent vectors will replicate each value with probability `similarity`.

# Arguments

  - `n_vecs::Integer`: Number of minivectors.
  - `n_values::Integer`: Number of features in each mini-vector.
  - `simiarlity::Number`: Degree of similarity between each mini-vector.

# Example

```jldoctest
julia> using Random; Random.seed!(1);

julia> vecs = create_vec(4, 4, 0.5)
4-element Vector{Vector{Float64}}:
 [0.0, -1.0, -1.0, 1.0]
 [0.0, -1.0, 1.0, 1.0]
 [0.0, -1.0, 1.0, 1.0]
 [0.0, 1.0, 0.0, 1.0]

julia> vecs[1]'vecs[2] / length(vecs[1]) # dot product
0.25
```
"""
function create_vec(n_vecs::Integer, n_values::Integer, similarity::Number)
    ref = create_vec(n_values)
    out = [ref]
    for _ in 2:n_vecs
        push!(out, trace_replication(ref,similarity))
    end
    return out
end


"""
    trace_replication(trace::Vector{Number}, similarity::Number, decay::Bool=false)

Returns a degraded replication of a given `trace` vector. The elements of the returned
vector are chosen from the original with a probabiliy of `similarity`. Values not
copied from the original vector are randomly samped from `[-1.0 0.0 1.0]` or assigned
`0.0` if you are decaying the trace (`decay==true`).

# Arguments

  - `trace::Vector{Number}`: Vector of integers to be duplicated.
  - `similarity::Number`: Degree of similarity between the original trace and resulting vector.
  - `decay::Bool`: Indicates if the `similarity` argument should be treated as the degree of decay. If `true`, then `similarity` is inverted, i.e., subtraced from 1.

# Example

```jldoctest
julia> using Random; Random.seed!(1);

julia> vector = [1.0, 1.0, 0.0, -1.0];

julia> trace_replication(vector, 0.5)
4-element Vector{Float64}:
 1.0
 1.0
 0.0
 0.0
```
"""
function trace_replication(trace::Vector{<:Number}, similarity::Number, decay::Bool=false)
    out = zeros(length(trace))
    for i in eachindex(trace)
        if trace[i] != 0.0
            if rand() > similarity
                out[i] = decay ? 0.0 : rand([-1.0,0.0,1.0])
            else
                out[i] = deepcopy(trace[i])
            end
        end
    end
    return out
end


"""
    trace_replication(trace::Information, similarity::Number, decay::Bool=false)
    trace_replication(trace::Trace, similarity::Number, decay::Bool=false)

Returns a degraded replication of a given object under the [HypothesisGeneration`](@ref) 
hierarchy. The elements of the minivectors within the returned object are chosen 
from the original minivectors with a probabiliy of `similarity`. Values not
copied from the original vector are randomly samped from `[-1.0 0.0 1.0]`or 
assigned `0.0` if you are decaying the trace (`decay==true`)

The following [`HypothesisGeneration`](@ref) types can be used as the [`Trace`](@ref) object:
```
Trace
├─ Information
│  └─ Context
├─ MemoryTrace
└─ ObsTrace
```

# Arguments

  - `trace`: Object of either [`Information`](@ref) or [`Trace`](@ref) to be replicated.
  - `similarity::Number`: Degree of similarity between the original trace and resulting object.
  - `decay::Bool`: Indicates if the `similarity` argument should be treated as the degree of decay. If `true`, then `similarity` is inverted, i.e., subtraced from 1.

# Example

```jldoctest
julia> using Random; Random.seed!(1);

julia> context = Context(:a, "this is context a", 5)
Context
  label: Symbol a
  description: String "this is context a"
  n_features: Integer 5
  contents: Array{Float64}((5,)) [-1.0, 0.0, 1.0, 0.0, 1.0]
  A_i: Float64 0.0

julia> trace_replication(context, 0.5)
Context
  label: Symbol a
  description: String "this is context a"
  n_features: Integer 5
  contents: Array{Float64}((5,)) [-1.0, 0.0, 1.0, 0.0, -1.0]
  A_i: Float64 0.0
```

"""
function trace_replication(trace::Information, similarity::Number, decay::Bool=false)
    new_context = deepcopy(trace)
    new_feature_vec = trace_replication(deepcopy(trace.contents), similarity, decay)
    new_context.contents = new_feature_vec
    return new_context
end
function trace_replication(trace::Trace, similarity::Number, decay::Bool=false)
    new_trace = deepcopy(trace)
    for info in new_trace.contents
        new_contents = trace_replication(deepcopy(info.contents), similarity, decay)
        info.contents = new_contents
    end
    return new_trace
end


@doc raw"""
    trace_similarity(probe::Vector{<:Number}, trace::Vector{<:Number})

Calculates the similarity between a probe vector and a trace. Eq 1 in Thomas et al. (2008) [^1].
Essentially a dot product.

```math
\begin{equation}
    S_i = \frac{\sum_{j=1}^n P_j T_{i,j}}{\sqrt{\sum_{j=1}^n P_j^2} \sqrt{\sum_{j=1}^n T_{i,j}^2}}
\end{equation}
```

# Arguments

  - `probe::Vector{<:Number}`: A vector of integers to be compared to the `trace`.
  - `trace::Vector{<:Number}`: A vector of integers.

# Example

```jldoctest
julia> a = [1.0, -1.0, 0.0, 1.0];

julia> b = [1.0, 0.0, 0.0, 1.0];

julia> trace_similarity(a,b)
0.6666666666666666
```

[^1]: Thomas, R. P., Dougherty, M. R., Sprenger, A. M., & Harbison, J. (2008). Diagnostic hypothesis generation and human judgment. _Psychological Review, 115_(1), 155-185. https://psycnet.apa.org/doi/10.1037/0033-295X.115.1.155
"""
function trace_similarity(probe::Vector{<:Number}, trace::Vector{<:Number})
    length(probe) == length(trace) || throw(DimensionMismatch("the lengths of the probe (n=$(length(probe))) and trace (n=$(length(trace))) vectors must be the same!"))
    N = length(probe)
    for (p,t) in zip(probe,trace)
        (p .== 0.0) && (t .== 0.0) ? N -= 1 : nothing
    end
    return (probe'trace) / float(N)
end


@doc raw"""
    trace_similarity(probe::Information, trace::Information)
    trace_similarity(probe::Trace, trace::Trace)

Calculates the similarity between a probe [`HypothesisGeneration`](@ref) object and a trace object. 
If there are multiple [`Context`](@ref) vectors in the `probe` and `trace`, it will recursively
compute [`trace_similarity`](@ref) for each vector and apply it to the overall `S_i` value.

```math
\begin{equation}
    S_i = \frac{\sum_{j=1}^n P_j T_{i,j}}{\sqrt{\sum_{j=1}^n P_j^2} \sqrt{\sum_{j=1}^n T_{i,j}^2}}
\end{equation}
```

The following types are accepted as `probe` and `trace` objects:

```
Trace
├─ Information
│  └─ Context
├─ MemoryTrace
└─ ObsTrace
```

# Arguments

  - `probe`: An object of either [`Information`](@ref) or [`Trace`](@ref) to be compared to `trace`.
  - `trace`: An object of either [`Information`](@ref) or [`Trace`](@ref).

# Example

```jldoctest
julia> using Random; Random.seed!(25);

julia> context_a = Context(:a, "this is context a", 10);

julia> context_b = trace_replication(context_a, 0.5);

julia> trace_similarity(context_a, context_b)
0.6

```
"""
function trace_similarity(probe::Information, trace::Information)
    return trace_similarity(probe.contents, trace.contents)
end
function trace_similarity(probe::Trace, trace::Trace)  
    # Extract contents from both vectors
    probe_contents = getfield.(probe.contents, :contents)
    trace_contents = getfield.(trace.contents, :contents)
    # Flatten contents and return trace similarity
    return trace_similarity(vcat(probe_contents...), vcat(trace_contents...))
end


@doc raw"""
    trace_activation(probe::Vector, trace::Vector, exponent::Integer=3)
    trace_activation(probe::HypothesisGeneration, trace::HypothesisGeneration, exponent::Integer=3)

The [`trace_similarity`](@ref) calculation accelerated by an exponent of degree `exponent`, usually 3.
This is analogous to a cubed cosine similarity. See Eqs 1 and 2 in Thomas et al. (2008) [^1].

```math
\begin{equation}
    A_i = \left( \frac{\sum_{j=1}^n P_j T_{i,j}}{\sqrt{\sum_{j=1}^n P_j^2} \sqrt{\sum_{j=1}^n T_{i,j}^2}} \right) ^3
\end{equation}
```

See the [`trace_similarity`](@ref) documentation for the base operation.

# Arguments

  - `probe`: An object to be compared to the `trace`.
  - `trace`: An object to be compared to the `probe`.
  - `exponent::Integer=3`: Degree of acceleration to the `S_i` value.

# Example

```jldoctest
julia> using Random; Random.seed!(25);

julia> context_a = Context(:a, "this is context a", 10);

julia> context_b = trace_replication(deepcopy(context_a), 0.5);

julia> trace_activation(context_a, context_b)
0.216
```

"""
function trace_activation(probe::Vector{<:Number}, trace::Vector{<:Number}, exponent::Integer=3)
    return trace_similarity(probe, trace) ^ exponent
end
function trace_activation(probe::HypothesisGeneration, trace::HypothesisGeneration, exponent::Integer=3)
    return trace_similarity(probe,trace) ^ exponent
end


"""
    trace_decay!(trace::Information, decay::Number)
    trace_decay!(trace::Trace, decay::Number)
    trace_decay!(store::MemoryStore, decay::Number)

Returns a [`HypothesisGeneration`](@ref) object with constituent [`Information`](@ref) vectors of numbers that have been
copied to the new object with probability `1 - decay`.

# Arguments

  - `trace`: An [`Information`](@ref), [`Trace`](@ref), or [`MemoryStore`](@ref) object with contents to be degraded.
  - `decay::Number`: The degree of decay to the `trace` elements.

# Example

```jldoctest
julia> using Random; Random.seed!(2);

julia> context_a = Context(:a, "this is context a", 10);

julia> context_b = deepcopy(context_a);

julia> trace_decay!(context_b, 0.5);

julia> trace_similarity(context_a, context_b)
0.6
```
"""
function trace_decay!(trace::Information, decay::Number)
    trace.contents = trace_replication(trace.contents, 1.0 - decay, true)
end
function trace_decay!(trace::Trace, decay::Number)
    for info in trace.contents
        trace_decay!(info, 1.0 - decay)
    end
end
function trace_decay!(store::MemoryStore, decay::Number)
    for trace in store.contents
        trace_decay!(trace, 1.0 - decay)
    end
end


"""
    bounce_memory!(observation::Trace, memory::MemoryStore)

Runs the [`trace_activation`](@ref) command between the `observation` object and all
objects in the `memory` object. Activation values `A_i` and then written back
to the [`Information`](@ref) vectors in `memory`.

This operation is used in parallel with [`echo_content`](@ref) and [`SetofContenders`](@ref)
functions.

# Arguments

  - `observation::Trace`: Observation from the real world to be compared to memory.
  - `memory::MemoryStore`: Memory store that the `observation` is compared to.

"""
function bounce_memory!(observation::Trace, memory::MemoryStore)
    for content in memory.contents
        content.A_i = trace_activation(observation, content)
    end
end


"""
    clear_content!(context::Information)
    clear_content!(trace::Trace)

Resets [`Information`](@ref) vectors and activation (`A_i`) values back to zeros. No other
fields, such as `description` are affected. Mostly useful for running multiple 
simulations.

# Arguments

  - An [`Information`](@ref) or [`Trace`](@ref) object to be reset.

# Example

```jldoctest
julia> using Random; Random.seed!(10);

julia> context_a = Context(:a, "this is context a", 4)
Context
  label: Symbol a
  description: String "this is context a"
  n_features: Int64 4
  contents: Array{Float64}((4,)) [-1.0, -1.0, -1.0, -1.0]
  A_i: Float64 0.0

julia> clear_content!(context_a);

julia> context_a
Context
  label: Symbol a
  description: String "this is context a"
  n_features: Int64 4
  contents: Array{Float64}((4,)) [0.0, 0.0, 0.0, 0.0]
  A_i: Float64 0.0
```
"""
function clear_content!(context::Information)
    context.contents = zeros(length(context.contents))
end
function clear_content!(trace::Trace)
    for context in trace.contents
        clear_content!(context)
    end
    trace.A_i = 0.0
end


"""
    standardize_trace!(trace::Trace)

Divides elements of all [`Information`](@ref) objects by the largest abolute value across all
objects, restricting values in the range of `[-1.0, 1.0]`. Helps resolve the _ambiguous 
recall problem_ in global memory models (Hintzman, 1986) [^2].

# Arguments

  - `trace`: a [`Trace`](@ref) object with unstandardized [`Context`](@ref) mini-vectors.

```jldoctest
julia> using Random; Random.seed!(5);

julia> context_a = Context(:a, "this is context a", 5);

julia> context_a.contents .*= rand(5)
5-element Vector{Float64}:
  0.4113527070924652
  0.8090056332338996
 -0.4913697257138979
  0.0
  0.0

julia> trace_a = MemoryTrace(:event_a, "this is event a", [context_a]);

julia> standardize_trace!(trace_a);

julia> trace_a.contents[1].contents
5-element Vector{Float64}:
  0.5084670491701446
  1.0
 -0.6073749125203349
  0.0
  0.0
```

[^2]: Hintzman, D. L. (1986). "Schema abstraction" in a multiple-trace memory model. _Psychological Review, 93_(4), 411-428. https://psycnet.apa.org/doi/10.1037/0033-295X.93.4.411
"""
function standardize_trace!(trace::Trace)
    # Find the largest absolute value in all contexts
    val = vcat([deepcopy(context.contents) for context in trace.contents]...)
    val = maximum(abs.(val))
    # Iterate through contexts and divide by the value
    for context in trace.contents
        context.contents ./= val
    end
end


@doc raw"""
    echo_content(observation::Trace, memory::MemoryStore, 
                 conditional::Bool=true, standardize::Bool=true)

Computes [`trace_activation`](@ref) between the `observation` object and all objects in
the `memory` store. [`Information`](@ref) vectors that are above activation threshold
`memory.A_c` are then multiplied against their conditional activation values `A_i`
(if `conditional==true`) and then summed across columns. The result is the "echo content" 
from a given memory store, i.e., a [`MemoryTrace`](@ref) object with float values in the
[`Information`](@ref) objects. The `standardize` parameter controls whether the operation
controls for Hintzman's (1986) [^2] _ambiguous recall problem_.

The resulting conditional echo content is given by Eq 7 in Thomas et al. (2008) [^1]:

```math
\begin{equation}
    C_c = \sum_{i=1}^K A_i T_{i,j}
\end{equation}
```

# Arguments

  - `observation::Trace`: Observation made in the real world, usually a copy of a prototype.
  - `memory::MemoryStore`: A [`MemoryStore`](@ref) object by which the `observation` is "bounced" off of.
  - `conditional::Bool=true`: Flag that indicates whether or not the program should calculate conditional echo content. `true` by default.
  - `standardized::Bool=true`: Flat that indicates whether or not the echo content should be standardized.

"""
function echo_content(observation::Trace, memory::MemoryStore, 
                      conditional::Bool=true, standardize::Bool=true)
    # Get activation values first 
    bounce_memory!(observation, memory)

    # Initialize a blank memory trace
    echo = deepcopy(memory.contents[1])
    clear_content!(echo)

    # Iterate through above-threshold traces
    for trace in memory.contents
        if trace.A_i > memory.A_c
            # Iterate through contexts
            for context in trace.contents
                trace_context = deepcopy(context.contents)
                conditional ? (trace_context .*= trace.A_i) : nothing
                filter(n -> n.label == context.label, echo.contents)[1].contents += trace_context
            end
        end
    end
    # Standardize if requested
    standardize ? standardize_trace!(echo) : nothing
    # Clean up echo trace and return
    echo.label = :echo
    echo.description = "echo content"
    return echo
end


@doc raw"""
    populate_soc!(echo::MemoryTrace, soc::SetofContenders, semantic::SemanticMemory)
    populate_soc!(echo::MemoryTrace, model::HyGeneModel)

This operation compares a trace (`echo`) to contents in semantic memory (`semantic`) 
and writes semantic traces with activation values above a dynamic threshold (`soc.act_min`)
to the [`SetofContenders`](@ref) object. 

Semantic traces are randomly sampled, and those that are above threshold will get written 
to the [`SetofContenders`](@ref) and the activation threshold (`soc.act_min`) will be set to that 
trace's activation (`trace.A_i`). If a trace's activation value does not exceed the 
threshold, then it will be considered a retrieval failure and will increment the failure
count variable (`t`) by 1. 

The function will continue to randomly sample traces with replacement until the maximum
number of retrieval attempts (`t_max`) has been reached.

# Arguments

  - `echo::MemoryTrace`: Echo content trace to be compared to [`SemanticMemory`](@ref).
  - `soc::SetofContenders`: Working memory store holding the contenders.
  - `semantic::SemanticMemory`: [`SemanticMemory`](@ref) store with trace exemplars.
  - `model::HyGeneModel`: A generic [`Model`](@ref) object with a [`SetofContenders`](@ref) and [`SemanticMemory`](@ref) store.

"""
function populate_soc!(echo::MemoryTrace, soc::SetofContenders, semantic::SemanticMemory)
    # Make sure SOC parameters are reset
    soc.act_min = 0.0
    soc.contenders = Vector{MemoryTrace}()
    soc.n_contenders = 0
    soc.t = 0
    # First, bounce echo against semantic memory
    bounce_memory!(echo, semantic)
    # Next, extract activation values to serve as sample weights
    # Sanitize the values so that Julia doesn't get mad
    act_weights = getfield.(semantic.contents, :A_i)
    act_weights[act_weights .< 0.0] .= 0.0
    act_weights = map(x -> isnan(x) ? zero(x) : x, act_weights)
    act_weights ./ sum(act_weights)
    # Iterate through semantic memory
    while (soc.t < soc.t_max)   # While the model has not hit the threshold
        # Randomly select traces from semantic memory weighted by their activations
        #sm_item = sample(semantic.contents, Weights(act_weights))
        sm_item = sample(semantic.contents)
        # Make sure activation is higher than minimum
        if (sm_item.A_i > soc.act_min) && (sm_item.A_i > 0.0) && (!isnan(sm_item.A_i))
                # Make sure item is not already in SOC
                if sm_item.label in [content.label for content in soc.contenders] 
                    soc.t +=1
                else
                    push!(soc.contenders, deepcopy(sm_item))
                    soc.n_contenders +=1 
                    soc.act_min = sm_item.A_i
                end
        else
            soc.t += 1
        end
    end
end
function populate_soc!(echo::MemoryTrace, model::HyGeneModel)
    populate_soc!(echo, model.working_memory, model.semantic_memory)
end


@doc raw"""
    cond_echo_intensity(probe::MemoryTrace, memory::MemoryStore, conditional::Bool=true)

Returns the conditional echo intensity of a memory trace object (`probe`)
against a memory store (`memory`). Echo intensity values are computed against
the contents of the `probe` for each item in `memory`. All intensity values
above a threshold `A_c` are then summed and divided by the number of 
above-threshold activations to produce `I_c`. If `conditional` is set to
`false`, the function will return the sum of all intensity values divided
by the number of items with activations above zero.

As per Eq. 5 in Thomas et al. (2008) [^1]:

```math
\begin{equation}
    I_c = \frac{\sum I_{A_i \geq A_c}}{K}
\end{equation}
```

# Arguments

  - `probe`: A [`Trace`](@ref) object to be compared to the memory store.
  - `memory`: A [`MemoryStore`](@ref) object with traces used to generate echo intensity.
  - `conditional::Bool=true`: Flag to indicate whether the function should return conditional echo intensity. If `false`, then it will only return echo intensity and _not_ conditional echo intensity.

"""
function cond_echo_intensity(probe::MemoryTrace, memory::MemoryStore, conditional::Bool=true)
    # Bounce probe off of memory
    bounce_memory!(probe, memory)
    # Extract above-threshold activations
    acts = Vector([copy(trace.A_i) for trace in memory.contents])
    acts = acts[acts.>0.0]
    cond_acts = acts[acts.>memory.A_c]
    # Return echo intensity
    if conditional && (length(cond_acts) > 0)
        return sum(cond_acts) / length(cond_acts)
    elseif !conditional && (length(acts) > 0)
        return sum(acts) / length(acts)
    else
        return 0.0
    end
end

@doc raw"""
    soc_posterior_prob(soc, memory)
    soc_posterior_prob(model)

Returns a vector of posterior probabilities for all leading contenders in the
[`SetofContenders`](@ref). This is accomplished by computing conditional echo intensity
values for all contenders against [`LongTermMemory`](@ref) and dividing them by the sum.

As per Eq. 8 in Thomas et al. (2008) [^1]:

```math
\begin{equation}
    P(H_i | D_{obs}) = \frac{I_{C_i}}{\sum_{i=1}^w I_{C_i}}
\end{equation}
```

"""
function soc_posterior_prob(soc::SetofContenders, memory::MemoryStore)
    if soc.n_contenders == 0 
        @error "There are no leading contenders in the SoC."
    elseif soc.n_contenders == 1
        @warn "There is only 1 contender in the SoC; thus, the probability will always be 1.0."
        return [1.0]
    else
        echo_intensity = cond_echo_intensity.(soc.contenders, memory)
        return echo_intensity ./ sum(echo_intensity)
    end
end
soc_posterior_prob(model::HyGeneModel) = soc_posterior_prob(model.working_memory, model.long_term_memory)


"""
"""
function soc_winner(model::HyGeneModel)
    n_contenders = 0
    if model.working_memory.n_contenders > 0
        n_contenders = model.working_memory.n_contenders
        if n_contenders == 1
            post_prob = 1.0
            winner = model.working_memory.contenders[1]
        else
            post_prob = soc_posterior_prob(model.working_memory, model.long_term_memory)
            winner = model.working_memory.contenders[argmax(getfield.(model.working_memory.contenders, :A_i))]
        end
        winner_label = winner.label
        acc = (obs.label == winner.label)*1
    else
        winner = NaN
        winner_label = NaN
        acc = 0
    end
    return (winner, winner_label, acc)
end