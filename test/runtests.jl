using SafeTestsets

@safetestset "compute_activations" begin
    using HyGene
    using Test
    include("example_setup.jl")

    activations = compute_activations(traces, probe)

    true_activations =
        [0.2963, 1.0, 0.5787, 1.0, 1.0, 0.5787, 0.002, 0.0156, 0.0156, 0.0156]

    @test activations ≈ true_activations atol = 5e-4
end

@safetestset "compute_cond_echo_content" begin
    @safetestset "1" begin
        using HyGene
        using Test
        include("example_setup.jl")

        activations = compute_activations(traces, probe)
        threshold = 0.216

        true_cond_echos = [3.88, -1.88, 0.0, 0.0, 4.45, 4.45, -4.45, 4.45, 0.0]
        cond_echos = compute_cond_echo_content(activations, hypotheses, threshold)

        @test cond_echos ≈ true_cond_echos atol = 5e-1
    end

    @safetestset "1" begin
        using HyGene
        using Test
        include("example_setup.jl")

        activations = compute_activations(traces, probe)
        threshold = 0.216

        true_cond_echos = [3.88, -1.88, 0.0, 0.0, 4.45, 4.45, -4.45, 4.45, 0.0]
        cond_echos = compute_cond_echo_content(activations, traces, threshold)
        true_cond_echos = [0.0, 3.58, -4.45, 3.88, 0.0, 4.45, -4.45, 4.16, 0.0]

        @test cond_echos ≈ true_cond_echos atol = 5e-1
    end
end

@safetestset "make_unspecified_probe" begin
    using HyGene
    using Test
    include("example_setup.jl")

    hypotheses = stack(_hypotheses, dims = 2)

    activations = compute_activations(traces, probe)
    threshold = 0.216

    unspecified_probe = create_unspecified_probe(activations, traces, hypotheses, threshold)

    true_unspecified_probe = [
        0.0,
        0.8,
        -1.0,
        0.87,
        0.0,
        1.0,
        -1.0,
        0.93,
        0.0,
        0.87,
        -0.42,
        0.0,
        0.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        0.0
    ]

    @test unspecified_probe ≈ true_unspecified_probe atol = 1e-2
end

@safetestset "semantic activation" begin
    using HyGene
    using Test
    include("example_setup.jl")

    activations = compute_activations(traces, probe)
    threshold = 0.216

    unspecified_probe = create_unspecified_probe(activations, traces, hypotheses, threshold)

    semantic_activation = compute_activations(semantic_memory, unspecified_probe)

    true_semantic_activation = [0.7491, 0.0825, 0.0787]

    @test semantic_activation ≈ true_semantic_activation atol = 1e-4
end

@safetestset "retreival probabilities" begin
    using HyGene
    using HyGene: normalize
    using Test
    include("example_setup.jl")

    activations = compute_activations(traces, probe)
    threshold = 0.216

    unspecified_probe = create_unspecified_probe(activations, traces, hypotheses, threshold)

    semantic_activation = compute_activations(semantic_memory, unspecified_probe)

    retrieval_probs = normalize(semantic_activation)
    true_retrieval_probs = [0.8229, 0.0906, 0.0864]
    @test retrieval_probs ≈ true_retrieval_probs atol = 1e-4
end

@safetestset "compute_cond_echo_intensities" begin
    using HyGene
    using Test
    include("example_setup.jl")

    activations = compute_activations(traces, probe)
    threshold = 0.216

    semantic_probes = @view semantic_memory[10:end, :]
    echo_intensities =
        compute_cond_echo_intensities(activations, hypotheses, semantic_probes, threshold)

    #@test echo_intensities ≈ [0.742, 0.196, 0.267] atol = 1e-3
    @test echo_intensities ≈ [0.742, 0.196, 0.3548] atol = 1e-3
end

@safetestset "update_working_memory!" begin
    @safetestset "1" begin
        using HyGene
        using HyGene: update_working_memory!
        using Test
        include("example_setup.jl")

        model = HyGeneModel(;
            t_max = 5,
            κ = 4,
            ρ = 0.85,
            τ = 0.21,
            data_map = (; f1 = 1:9),
            hypothesis_map = (; h = 10:18),
            episodic_memory = [traces; hypotheses],
            semantic_memory,
            working_memory = Int[]
        )

        n_fails = 0
        semantic_activation = [0.7, 0.3]
        τₛ = 0.0
        idx = 2
        n_fails, τₛ = update_working_memory!(model, semantic_activation, idx, n_fails, τₛ)
        @test n_fails == 0
        @test τₛ == 0.30
        @test model.working_memory == [2]
    end

    @safetestset "2" begin
        using HyGene
        using HyGene: update_working_memory!
        using Test
        include("example_setup.jl")

        model = HyGeneModel(;
            t_max = 5,
            κ = 4,
            ρ = 0.85,
            τ = 0.21,
            data_map = (; f1 = 1:9),
            hypothesis_map = (; h = 10:18),
            episodic_memory = [traces; hypotheses],
            semantic_memory,
            working_memory = [2]
        )
        n_fails = 0
        semantic_activation = [0.7, 0.3]
        τₛ = 0.3
        idx = 1
        n_fails, τₛ = update_working_memory!(model, semantic_activation, idx, n_fails, τₛ)
        @test n_fails == 0
        @test τₛ == 0.30
        @test model.working_memory == [2, 1]
    end

    @safetestset "3" begin
        using HyGene
        using HyGene: update_working_memory!
        using Test
        include("example_setup.jl")

        model = HyGeneModel(;
            t_max = 5,
            κ = 4,
            ρ = 0.85,
            τ = 0.21,
            data_map = (; f1 = 1:9),
            hypothesis_map = (; h = 10:18),
            episodic_memory = [traces; hypotheses],
            semantic_memory,
            working_memory = [2]
        )
        n_fails = 0
        semantic_activation = [0.7, 0.3]
        τₛ = 0.3
        idx = 2
        n_fails, τₛ = update_working_memory!(model, semantic_activation, idx, n_fails, τₛ)
        @test n_fails == 1
        @test τₛ == 0.30
        @test model.working_memory == [2]
    end

    @safetestset "4" begin
        using HyGene
        using HyGene: update_working_memory!
        using Test
        include("example_setup.jl")

        model = HyGeneModel(;
            t_max = 5,
            κ = 1,
            ρ = 0.85,
            τ = 0.21,
            data_map = (; f1 = 1:9),
            hypothesis_map = (; h = 10:18),
            episodic_memory = [traces; hypotheses],
            semantic_memory,
            working_memory = [2]
        )
        n_fails = 0
        semantic_activation = [0.7, 0.3]
        τₛ = 0.3
        idx = 1
        n_fails, τₛ = update_working_memory!(model, semantic_activation, idx, n_fails, τₛ)
        @test n_fails == 0
        @test τₛ == 0.70
        @test model.working_memory == [1]
    end

    @safetestset "5" begin
        using HyGene
        using HyGene: update_working_memory!
        using Test
        include("example_setup.jl")

        model = HyGeneModel(;
            t_max = 5,
            κ = 1,
            ρ = 0.85,
            τ = 0.21,
            data_map = (; f1 = 1:9),
            hypothesis_map = (; h = 10:18),
            episodic_memory = [traces; hypotheses],
            semantic_memory,
            working_memory = [2]
        )
        n_fails = 1
        semantic_activation = [0.7, 0.3]
        τₛ = 0.3
        idx = 1
        n_fails, τₛ = update_working_memory!(model, semantic_activation, idx, n_fails, τₛ)
        @test n_fails == 0
        @test τₛ == 0.70
        @test model.working_memory == [1]
    end

    @safetestset "6" begin
        using HyGene
        using HyGene: update_working_memory!
        using Test
        include("example_setup.jl")

        model = HyGeneModel(;
            t_max = 5,
            κ = 3,
            ρ = 0.85,
            τ = 0.21,
            data_map = (; f1 = 1:9),
            hypothesis_map = (; h = 10:18),
            episodic_memory = [traces; hypotheses],
            semantic_memory,
            working_memory = Int[]
        )
        n_fails = 0
        semantic_activation = [0.4, 0.3, 0.15, 0.05]
        τₛ = 0
        idx = 3
        n_fails, τₛ = update_working_memory!(model, semantic_activation, idx, n_fails, τₛ)
        @test n_fails == 0
        @test τₛ == 0.15
        @test model.working_memory == [3]

        idx = 3
        n_fails, τₛ = update_working_memory!(model, semantic_activation, idx, n_fails, τₛ)
        @test n_fails == 1
        @test τₛ == 0.15
        @test model.working_memory == [3]

        idx = 4
        n_fails, τₛ = update_working_memory!(model, semantic_activation, idx, n_fails, τₛ)
        @test n_fails == 2
        @test τₛ == 0.15
        @test model.working_memory == [3]

        idx = 2
        n_fails, τₛ = update_working_memory!(model, semantic_activation, idx, n_fails, τₛ)
        @test n_fails == 0
        @test τₛ == 0.15
        @test model.working_memory == [3, 2]

        idx = 1
        n_fails, τₛ = update_working_memory!(model, semantic_activation, idx, n_fails, τₛ)
        @test n_fails == 0
        @test τₛ == 0.15
        @test model.working_memory == [3, 2, 1]
    end
end

@safetestset "populate_working_memory!" begin
    @safetestset "1" begin
        using HyGene
        using HyGene: populate_working_memory!
        using Test
        include("example_setup.jl")

        model = HyGeneModel(;
            t_max = 5,
            κ = 1,
            ρ = 0.85,
            τ = 0.21,
            data_map = (; f1 = 1:9),
            hypothesis_map = (; h = 10:18),
            episodic_memory = [traces; hypotheses],
            semantic_memory,
            working_memory = Int[]
        )

        semantic_activation = [0,0]
        populate_working_memory!(model, semantic_activation)
        @test isempty(model.working_memory)
    end

    @safetestset "2" begin
        using HyGene
        using HyGene: populate_working_memory!
        using Test
        include("example_setup.jl")

        model = HyGeneModel(;
            t_max = 5,
            κ = 1,
            ρ = 0.85,
            τ = 0.21,
            data_map = (; f1 = 1:9),
            hypothesis_map = (; h = 10:18),
            episodic_memory = [traces; hypotheses],
            semantic_memory,
            working_memory = Int[]
        )

        semantic_activation = [1,0]
        populate_working_memory!(model, semantic_activation)
        @test model.working_memory == [1]
    end

    @safetestset "2" begin
        using HyGene
        using HyGene: populate_working_memory!
        using Random 
        using Statistics
        using Test
        include("example_setup.jl")

        Random.seed!(5747)

        model = HyGeneModel(;
            t_max = 5,
            κ = 1,
            ρ = 0.85,
            τ = 0.21,
            data_map = (; f1 = 1:9),
            hypothesis_map = (; h = 10:18),
            episodic_memory = [traces; hypotheses],
            semantic_memory,
            working_memory = Int[]
        )

        semantic_activation = [1.0,0.0,1.0]
        results = fill(0, 1000)
        for i ∈ 1:1000
            populate_working_memory!(model, semantic_activation)
            @test length(model.working_memory) == 1
            results[i] = model.working_memory[1]
        end
        @test mean(results .== 1) ≈ .50 atol = .03
    end
end

@safetestset "replicate_trace" begin
    @safetestset "replicate_trace" begin
        using HyGene
        using Random 
        using Statistics
        using Test
    
        Random.seed!(62)
    
        trace = make_traces(100_000)
        # encoding fidelity
        ρ = 0.0
        new_trace = replicate_trace(trace, ρ)
        # 1/3 chance a feature in trace is zero
        # all features in new_trace are zero 
        @test 1/3 ≈ mean(trace .== new_trace) atol = .01
    end

    @safetestset "replicate_trace" begin
        using HyGene
        using Random 
        using Statistics
        using Test
    
        Random.seed!(62)
    
        trace = make_traces(100_000)
        # encoding fidelity
        ρ = .85
        new_trace = replicate_trace(trace, ρ)
        non_zero_idx = trace .≠ 0
        @test ρ ≈ mean(trace[non_zero_idx] .== new_trace[non_zero_idx]) atol = .005
        @test 2/3 ≈ mean(non_zero_idx) atol = .005
    end
end

@safetestset "encode" begin
    @safetestset "1" begin
        using HyGene
        using HyGene: encode
        using Random 
        using Statistics
        using Test
    
        Random.seed!(3323)
    
        # encoding fidelity
        ρ = .85
        encodings = [encode(1, ρ) for _ ∈ 1:100_000]
      
        @test ρ ≈ mean(encodings .== 1) atol = .005
    end

    @safetestset "2" begin
        using HyGene
        using HyGene: encode
        using Random 
        using Statistics
        using Test
    
        Random.seed!(414)
    
        # encoding fidelity
        ρ = .85
        encodings = [encode(0, ρ) for _ ∈ 1:100_000]
      
        @test 1 == mean(encodings .== 0)
    end
end