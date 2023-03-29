#=============================================================================================
HYGENE CALCULATION EXAMPLE

@author		: taylor-curley
@email		: tmc2737 "at" gmail.com
@date		: 03-27-23

NOTES		
    : Computations follow the example given in the Appendix of the original paper [^1]. 
    : Setting the seed should result in a single conditional echo intensity value of ~0.49
    : for the first ("correct") hypothesis.
    :
    
REFERENCES
    : [^1] Thomas, R. P., Dougherty, M. R., Sprenger, A. M., & Harbison, J. (2008). Diagnostic 
           hypothesis generation and human judgment. Psychological Review, 115(1), 155-185. 
           https://doi.org/10.1037/0033-295X.115.1.155

==============================================================================================#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Import Packages ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# use directory containing this file
cd(@__DIR__)
using Pkg 
# use package environment
Pkg.activate("..")
using HyGene, StatsBase, Random
# set seed
Random.seed!(100)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Define Hyperparameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
L = 0.85                            # Encoding fidelity parameter
n_features = 9                      # Number of features in a given mini-vector
t_max = 10                          # Max number of retrieval failures
ϕ = 4                               # Max number of items in WM
act_thresh = 0.216                  # Oddly-specific minimum activation from paper


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Specify Semantic Traces ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# First, generate generic hypotheses, both "data" and "hypothesis" components
hypothesis_labels = ["hypothesis 1 (E1)","hypothesis 2 (E2)","hypothesis 3 (E3)"]
hypotheses = Vector{Hypothesis}()
for h in hypothesis_labels
    push!(hypotheses, Hypothesis(h,n_features))
end

# Push data components to separate Contexts. Not needed, but an interesting exercise
context_labels = ["context 1 (E1)","context 2 (E2)","context 3 (E3)"]
contexts = Vector{Context}()
for i in 1:length(hypotheses)
    push!(contexts,Context(context_labels[i],length(hypotheses[i].data),hypotheses[i].data))
end


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Specify Episodic Traces ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Replicate E1 6 times, E2 3 times, and E3 1 time as observations
observations = Vector{Observation}()
for i in 1:6
    push!(observations, generate_observation("obs "*string(i)*" (E1)", hypotheses[1]))
end
for j in 1:3
    push!(observations, generate_observation("obs "*string(j+6)*" (E2)", hypotheses[2]))
end
push!(observations, generate_observation("obs 10 (E3)", hypotheses[3]))

# Generate traces with similarity = 0.85
traces = Vector{Trace}()
for obs in observations
    new = trace_replicator(obs,L)
    push!(traces, Trace(new.label, new.n_values, new.n_contexts, 0, new.data, new.hypothesis))
end


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Construct Simulation Object ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
params = (decay = 1-L, n_values = n_features, t_max = t_max, ϕ = 4, act_thresh = act_thresh)
ex_sim = HyGeneModel(;params...)
ex_sim.contexts = contexts
ex_sim.semantic_memory = hypotheses
ex_sim.long_term_memory = traces


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Run Simulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Generate an observation that is similar to E1 with blank hypothesis
probe = Observation("probe", n_features, 1, hypotheses[1].data, [])

# Compute trace activations in long term memory and find those above the threshold
trace_acts = []
for memory in ex_sim.long_term_memory
    push!(trace_acts, trace_activation(probe.data,memory.data))
end
where_trace_acts = findall(trace_acts .> ex_sim.act_thresh)

# Calculate the unspecified probe
probe_data = getfield.(ex_sim.long_term_memory,:data) .* trace_acts
probe_data = sum(probe_data[where_trace_acts])
probe_data ./= maximum(abs.(probe_data))
probe_hyp = getfield.(ex_sim.long_term_memory,:hypothesis).* trace_acts
probe_hyp = sum(probe_hyp[where_trace_acts])
probe_hyp ./= maximum(abs.(probe_hyp))

# Calculate activations of semantic memory
semantic_acts = []
for sem in ex_sim.semantic_memory
    probe = vcat(probe_data, probe_hyp)
    trace = vcat(sem.data, sem.hypothesis)
    sem_act = trace_activation(probe,trace)
    sem_act > 0.0 ? push!(semantic_acts,sem_act) : push!(semantic_acts,0.0)
end
normed_semantic_acts = semantic_acts ./ sum(semantic_acts)

# Sample hypotheses from semantic memory with weights derived from
# normalized semantic activations
while ex_sim.t < ex_sim.t_max
    h = sample(collect(1:length(semantic_acts)), Weights(normed_semantic_acts))
    if semantic_acts[h] > ex_sim.act_min_h
        if ex_sim.semantic_memory[h] in ex_sim.working_memory
            ex_sim.t += 1
        else
            ex_sim.act_min_h = copy(semantic_acts[h])
            push!(ex_sim.working_memory, ex_sim.semantic_memory[h])
        end
    else
        ex_sim.t += 1
    end
end

# Calculate conditional echo intensities using contents of working memory
cond_echo_i = []
hyp_list = getfield.(ex_sim.long_term_memory, :hypothesis) .* trace_acts
for hyp in ex_sim.working_memory
    hyp_acts = [trace_activation(hyp.hypothesis,trace) for trace in hyp_list]
    i_c = sum(hyp_acts[where_trace_acts])/length(where_trace_acts)
    push!(cond_echo_i, i_c)
end