#=============================================================================================
HYGENE CALCULATION EXAMPLE

@author		: taylor.curley
@email		: taylor.curley 'at' afrl.af.mil
@date		: 03-27-23

NOTES		
    : Computations follow the example given in the Appendix of the original paper [^1]. 
    :
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Define Hyperparameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
L = 0.85                            # Encoding fidelity parameter
n_features = 9                      # Number of features in a given mini-vector
t_max = 10                          # Max number of retrieval failures
ϕ = 4                               # Max number of items in WM

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Specify Hypotheses ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
hypothesis_labels = ["hypothesis 1 (E1)","hypothesis 2 (E2)","hypothesis 3 (E3)"]
hypotheses = Vector{Hypothesis}()
for h in hypothesis_labels
    push!(hypotheses, Hypothesis(h,n_features))
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Construct Simulation Object ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
params = (decay = 1-L, n_values = n_features, t_max = t_max, ϕ = 4)
ex_sim = HyGeneModel(;params...)