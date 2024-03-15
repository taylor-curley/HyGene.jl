using Documenter, HyGene
push!(LOAD_PATH,"../src/")

makedocs(
    sitename="HyGene.jl",
    format = Documenter.HTML(prettyurls=false),
    pages = [
        "Home" => "index.md",
        "API" => [
            "Structures" => "structs.md",
            "Constructors" => "constrs.md",
            "Simulation Controllers" => "sim_controller.md",
            "General Functions" => "functs.md"
        ]    
    ]
)

# deploydocs(
#     repo = "github.com/taylor-curley/HyGene.jl.git",
#     versions = nothing,
# )