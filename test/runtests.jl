using HyGene
using Test

@testset "HyGene.jl" begin
    n_units = 10
    n_contexts = 3
    decay = 0.5
    # Test Hypothesis object
    new_hyp = Hypothesis("test", n_units)
    @test length(new_hyp.content) == n_units
    @test typeof(new_hyp) <: Hypothesis
    @test typeof(new_hyp) <: HypothesisGeneration
    @test trace_similarity(new_hyp,new_hyp) == 1.0
    # Test Context object
    new_context = Context("test", n_units)
    @test length(new_context.content) == n_units
    @test typeof(new_context) <: Context
    @test typeof(new_context) <: HypothesisGeneration
    @test trace_similarity(new_context,new_context) == 1.0
    # Test Observation
    new_obs = Observation("test", n_units, n_contexts,
                          generate_item(n_units,n_contexts))
    @test length(new_obs.content) == n_contexts
    @test typeof(new_obs) <: Observation
    @test typeof(new_obs) <: HypothesisGeneration
    # Test trace
    new_trace = obs_to_trace(new_obs, decay)
    @test length(new_trace.content) == n_contexts
    @test typeof(new_trace) <: Trace
    @test typeof(new_trace) <: HypothesisGeneration
    @test trace_similarity(new_obs, new_trace) ≈ decay atol=0.25
    # TODO: Test HyGeneSim objects and basic functions
end
