using Documenter
using Literate
using StochasticVehicleScheduling

ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"

md_dir = joinpath(@__DIR__, "src")
literate_dir = joinpath(dirname(@__DIR__), "docs", "src", "literate")
jl_files = readdir(literate_dir)  # joinpath(literate_dir, "tutorial.jl")

for jl_file in jl_files
    Literate.markdown(
        joinpath(literate_dir, jl_file), md_dir; documenter=true, execute=false
    )
end

DocMeta.setdocmeta!(
    StochasticVehicleScheduling,
    :DocTestSetup,
    :(using StochasticVehicleScheduling);
    recursive=true,
)

makedocs(;
    modules=[StochasticVehicleScheduling],
    authors="Léo Baty and contributors",
    repo="https://github.com/BatyLeo/StochasticVehicleScheduling.jl/blob/{commit}{path}#{line}",
    sitename="StochasticVehicleScheduling.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://BatyLeo.github.io/StochasticVehicleScheduling.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "math.md",
        "dataset.md",
        "algorithms.md",
        "Learning with InferOpt.jl" => ["inferopt.md", "paper.md"],
        "api.md",
    ],
)

deploydocs(; repo="github.com/BatyLeo/StochasticVehicleScheduling.jl", devbranch="main")
