using StochasticVehicleScheduling
using Documenter

DocMeta.setdocmeta!(StochasticVehicleScheduling, :DocTestSetup, :(using StochasticVehicleScheduling); recursive=true)

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
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/BatyLeo/StochasticVehicleScheduling.jl",
    devbranch="main",
)
