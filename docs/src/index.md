![](assets/header.png)

## Overview

This package simulates the HyGene (**Hy**pothesis **Gene**ration) cognitive model. HyGene is suited for simulating memory-based decisions in the face of experience ("traces") and current information ("observations"). 

The code is based off of Thomas et al.'s [^1] original and subsequent [^2] papers, which itself is influenced by Hintzman's MINERVA2 architecture [^3] [^4] [^5].

## Installation

`HyGene.jl` is not yet available via the Julia package manager. For now, you will need to: 
  1. Clone the repository from Github, 
  2. Create a new `.jl` file in the `./model/` directory, and 
  3. Instantiate the package from the source code.

At the top of your new file, add the following code at the top of the page:

```julia
cd(@__DIR__)
using Pkg 
Pkg.activate("..")
using HypothesisGeneration
```

After this, you may add code to build a new `HyGene` model.

!!! note

    _Easier implementations will be added later on._

## Quick Example

The example below will show how to accomplish some basic tasks:

  1. Build a `HyGeneModel` object with **4 hypotheses**, each with **5 observations** and each observation containing **2** contexts,
  2. Create an outward observation based on the first (focal) prototype with the proportion of shared elements being `0.9`, 
  3. "Bounce" the observation off of long-term memory to create conditional echo content, and
  4. Determine the most likely hypothesis using the Set of Contenders.

```julia
# Setting seed to replicate results
using HypothesisGeneration, Random
Random.seed!(10)

# Create new HyGeneModel object
n_contexts = 2; n_hypotheses = 4; n_obs_per_hyp = 5;
new_mod = HyGeneModel(n_contexts, n_hypotheses, n_obs_per_hyp);

# Create observation with 0.9 fidelity (d_obs_sim)
d_obs_sim = 0.9;
obs = create_observation(new_mod.prototypes[1], d_obs_sim);

# Generate conditional echo content
echo = echo_content(obs, new_mod.long_term_memory);

# Find the best-fitting hypothesis in semantic memory
# Get the "winning" trace, label, and accuracy
populate_soc!(echo, new_mod);
winner_trace, winner, acc = soc_winner(new_mod);
```

When we look at the `winner`, we can see that the model found the correct hypothesis, i.e., `:hypothesis_1`:

```julia
julia> winner
:hypothesis_1
```

We can further probe the strength of this decision by looking at elements of the `SetofContenders`. For example, if we type `new_mod.working_memory.n_contenders` in the terminal, we will we see that the model only found one (`1`) contender before terminating search. Additionally, we can find the activation of the hypothesis given the echo by typing in `new_mod.working_memory.act_min`, which is around `0.358`.

The body of the documentation describes further functionality and boundaries on them. In the above example, we would not be able to calculate the posterior probability of the contender in the `SetofContenders`:

```julia
julia> soc_posterior_prob(new_mod)
┌ Warning: There is only 1 contender in the SoC; thus, the probability will always be 1.0.
└ @ HyGene d:\Documents\GitHub\HyGene.jl\src\utilities.jl:685
1-element Vector{Float64}:
 1.0
```

As you can see by the error message, if there is only one contender in the `SetofContenders`, such as in our example, then the posterior probability calculation does not work and defaults to `1.0`.

## References

[^1]: Thomas, R. P., Dougherty, M. R., Sprenger, A. M., & Harbison, J. (2008). Diagnostic hypothesis generation and human judgment. _Psychological Review, 115_(1), 155-185. https://psycnet.apa.org/doi/10.1037/0033-295X.115.1.155

[^2]: Thomas, R., Dougherty, M. R., & Buttaccio, D. R. (2014). Memory constraints on hypothesis generation and decision making. _Current Directions in Psychological Science, 23_(4), 264–270. https://doi.org/10.1177/0963721414534853 

[^3]: Hintzman, D. L. (1986). "Schema abstraction" in a multiple-trace memory model. _Psychological Review, 93_(4), 411-428. https://psycnet.apa.org/doi/10.1037/0033-295X.93.4.411

[^4]: Hintzman, D. L. (1984). MINERVA 2: A simulation model of human memory. _Behavior Research Methods, Instruments, & Computers, 16_(2), 96-101. https://doi.org/10.3758/BF03202365

[^5]: Hintzman, D. L. (1988). Judgments of frequency and recognition memory in a multiple-trace memory model. _Psychological Review, 95_(4), 528–551. https://doi.org/10.1037/0033-295X.95.4.528