using HyGene
using Test

@testset "HyGene.jl" begin
    # Specify simulation parameters
    n_units = 10
    n_contexts = 4
    n_hypotheses = 2
    n_traces = 20
    decay = 0.5
    # Test Hypothesis object
    new_hyp = Hypothesis("test", n_units)
    @test length(new_hyp.data) == n_units
    @test typeof(new_hyp) <: Hypothesis
    @test typeof(new_hyp) <: HypothesisGeneration
    @test trace_similarity(new_hyp,new_hyp) == 1.0
    # Test Context object
    new_context = Context("test", n_units)
    @test length(new_context.data) == n_units
    @test typeof(new_context) <: Context
    @test typeof(new_context) <: HypothesisGeneration
    @test trace_similarity(new_context,new_context) == 1.0
    # Test Observation
    new_obs = Observation("test", n_units, n_contexts,
                          generate_item(n_units,n_contexts))
    @test length(new_obs.data) == n_contexts
    @test typeof(new_obs) <: Observation
    @test typeof(new_obs) <: HypothesisGeneration
    # Test Trace
    new_trace = obs_to_trace(new_obs, decay)
    @test length(new_trace.data) == n_contexts
    @test typeof(new_trace) <: Trace
    @test typeof(new_trace) <: HypothesisGeneration
    @test trace_similarity(new_obs, new_trace) ≈ decay atol=0.25
    # Test HyGeneSim object
    vars = (n_values = n_units, n_contexts = n_contexts, n_lt_memory = n_traces, n_sem_memory = n_hypotheses, decay = decay)
    contexts = Vector{Context}([Context("context_"*string(i), n_units) for i in 1:n_contexts])
    hypotheses = Vector{Hypothesis}([Hypothesis("hypothesis_"*string(j), n_units) for j in 1:n_hypotheses])
    observations = Vector{Observation}(undef,0)
    traces = Vector{Trace}(undef,0)
    for i in 1:n_traces
        if rand() < 0.5
            cont = contexts[1:2]
            hyp = hypotheses[1]
        else
            cont = contexts[3:4]
            hyp = hypotheses[2]
        end
        obs = generate_observation("trace_"*string(i), cont, hyp)
        @test typeof(obs) <: Observation
        push!(observations, obs)
        trc = obs_to_trace(obs, decay)
        @test typeof(trc) <: Trace
        push!(traces, trc)
    end
    model = HyGeneModel(; vars..., 
                        contexts = deepcopy(contexts), 
                        semantic_memory = deepcopy(hypotheses),
                        long_term_memory = deepcopy(traces))
    @test typeof(model) <: HyGeneModel
    @test typeof(model) <: HypothesisGeneration
    @test length(model.long_term_memory) == n_traces
    @test length(model.semantic_memory) == n_hypotheses
    # Test simulation controllers
    @test trace_activation(traces[1],observations[1]) ≈ trace_similarity(traces[1],observations[1])^3 atol = 0.01
    @test trace_similarity(model.long_term_memory[1],observations[1]) ≈ decay atol=0.25
    @test echo_intensity(traces[1],model.long_term_memory) > 1.0
    trace_decay!(model)
    @test model.long_term_memory[1] != traces[1]
end
