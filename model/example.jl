###############################################################################################
#                                       Import Packages                                       #
###############################################################################################
# use directory containing this file
cd(@__DIR__)
using Pkg 
# use package environment
Pkg.activate("..")
using HyGene, StatsBase, Random


###############################################################################################
#                                        Simple Example                                       #
###############################################################################################
# Construct new simulation object and define some parameters
sim = HyGeneModel()         # Blank simulation
sim.decay = 0.2             # 0.2  probability of degradation
sim.n_values = 15           # 15 values in each minivector
sim.n_contexts = 2          # 2 context minivectors (not including the hypothesis)

# Construct hypotheses (e.g., seasonal illnesses)
hypothesis_labels = ["cold","flu"]
hypotheses = [Hypothesis(x,sim.n_values) for x in hypothesis_labels]

# Construct contexts (e.g., symptoms)
context_labels = ["cough","sneeze","fever","aches"]
contexts = [Context(x,sim.n_values) for x in context_labels]

# Construct the following observations using 2 contexts:
#       ["fever","cough"] -> ["flu"]    x2
#       ["fever","aches"] -> ["flu"]    x2
#       ["sneeze","aches"] -> ["flu"]   x2
#       ["sneeze","cough"] -> ["cold"]  x3
#       ["cough","sneeze"] -> ["cold"]  x3
obs_vecs1 = [[3,1,2,"flu1"],
             [3,4,2,"flu2"],
             [2,4,2,"flu3"]]
obs_vecs2 = [[2,1,1,"cold1"],
             [1,2,1,"cold2"]]
obs_vecs = vcat(repeat(obs_vecs1,2), repeat(obs_vecs2,3))
observations = []           # Instantiate observation list
for i in 1:length(obs_vecs) # Loop through observation vectors
    obs = obs_vecs[i]
    vecs = Vector{<:Real}[] # Strict definition seems to be needed for the moment
    for j in 1:2            # Loop through contexts
        push!(vecs,contexts[obs[j]].content)
    end
    push!(vecs, hypotheses[obs[3]].content) # Add "correct" hypothesis
    push!(observations, Observation(obs[4],sim.n_values,sim.n_contexts,vecs))
end

# Write observations to traces with decay
traces = obs_to_trace.(observations,sim.decay)

# Append simulation object
sim.contexts = contexts
sim.semantic_memory = hypotheses
sim.long_term_memory = traces

# Define new observation with blank hypothesis
#       ["fever","aches"] -> [?]
new_content = [contexts[3].content,contexts[4].content,zeros(sim.n_values)]
new_obs = Observation("new",sim.n_values,sim.n_contexts,new_content)

# Return echo intensity
new_obs_I = echo_intensity(new_obs, sim.long_term_memory)
