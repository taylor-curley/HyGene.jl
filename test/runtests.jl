using SafeTestsets

# a general test to make sure nothing breaks
# more tests to follow
@safetestset "smoke test" begin
    using HyGene
    using StatsBase
    using Test

    # LTM EXAMPLE: 4 types of unique events w/ 5 instances of each.
    # Each LTM trace consists of 4 information minivectors with 10
    # features each. Repeated instances have a similarity of 0.75
    context_a = Context(:a, "this is context a", 10)
    context_b = Context(:b, "this is context b", 10)
    context_c = Context(:c, "this is context c", 10)
    context_d = Context(:d, "this is context d", 10)
    context_vec = Vector{Context}([context_a, context_b, context_c, context_d])
    event_a = MemoryTrace(:event_a, "this is event a", deepcopy(context_vec))
    event_a_repl = Vector{MemoryTrace}([trace_replication(event_a, 0.75) for _ = 1:5])

    context_e = Context(:a, "this is context e", 10)
    context_f = Context(:b, "this is context f", 10)
    context_g = Context(:c, "this is context g", 10)
    context_h = Context(:d, "this is context h", 10)
    context_vec2 = Vector{Context}([context_e, context_f, context_g, context_h])
    event_b = MemoryTrace(:event_b, "this is event b", deepcopy(context_vec2))
    event_b_repl = Vector{MemoryTrace}([trace_replication(event_b, 0.75) for _ = 1:5])

    context_i = Context(:a, "this is context i", 10)
    context_j = Context(:b, "this is context j", 10)
    context_k = Context(:c, "this is context k", 10)
    context_l = Context(:d, "this is context l", 10)
    context_vec3 = Vector{Context}([context_i, context_j, context_k, context_l])
    event_c = MemoryTrace(:event_c, "this is event c", deepcopy(context_vec3))
    event_c_repl = Vector{MemoryTrace}([trace_replication(event_c, 0.75) for _ = 1:5])

    context_m = Context(:a, "this is context m", 10)
    context_n = Context(:b, "this is context n", 10)
    context_o = Context(:c, "this is context o", 10)
    context_p = Context(:d, "this is context p", 10)
    context_vec4 = Vector{Context}([context_m, context_n, context_o, context_p])
    event_d = MemoryTrace(:event_d, "this is event d", deepcopy(context_vec4))
    event_d_repl = Vector{MemoryTrace}([trace_replication(event_d, 0.75) for _ = 1:5])

    traces = vcat(event_a_repl, event_b_repl, event_c_repl, event_d_repl)
    ltm = LongTermMemory(traces, 0.1)
    sm = SemanticMemory(ltm)
    soc = SetofContenders(4)

    obs_contexts1 = Vector{Context}([context_a, context_b, context_d, context_l])
    observation1 = ObsTrace(:obs_a, "this is an observation", deepcopy(obs_contexts1))
    obs_contexts2 = Vector{Context}([context_a, context_b, context_e, context_l])
    observation2 = ObsTrace(:obs_b, "this is an observation", deepcopy(obs_contexts2))

    echo1 = echo_content(observation1, ltm)
    populate_soc!(echo1, soc, sm)
    soc_posterior_prob(soc, ltm)

    echo2 = echo_content(observation2, ltm)


    using Random
    Random.seed!(25)
    context_a = Context(:a, "this is context a", 10)
    context_b = trace_replication(context_a, 0.5)
    trace_similarity(context_a, context_b)

    using Random
    Random.seed!(25)
    context_a = Context(:a, "this is context a", 10)
    context_b = trace_replication(context_a, 0.5)
    trace_activation(context_a, context_b)
    @test true
end
