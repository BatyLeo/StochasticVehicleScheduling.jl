using Documenter
using Literate
using StochasticVehicleScheduling

DocMeta.setdocmeta!(
    StochasticVehicleScheduling,
    :DocTestSetup,
    :(using StochasticVehicleScheduling);
    recursive=true,
)

literate_dir = joinpath(dirname(@__DIR__), "docs", "src", "literate")
jl_file = joinpath(literate_dir, "tutorial.jl")
md_dir = joinpath(@__DIR__, "src")
Literate.markdown(jl_file, md_dir; documenter=true, execute=false)

makedocs(;
    modules=[StochasticVehicleScheduling],
    authors="BatyLeo and contributors",
    repo="https://github.com/BatyLeo/StochasticVehicleScheduling.jl/blob/{commit}{path}#{line}",
    sitename="StochasticVehicleScheduling.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://BatyLeo.github.io/StochasticVehicleScheduling.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md", "math.md", "dataset.md", "tutorial.md", "API" => "api.md"],
)

deploydocs(; repo="github.com/BatyLeo/StochasticVehicleScheduling.jl", devbranch="main")
