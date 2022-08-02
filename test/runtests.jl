using Aqua
using JuliaFormatter
using StochasticVehicleScheduling
using Test

format(StochasticVehicleScheduling; verbose=true)

include("subfiles/utils.jl")

@testset verbose = true "StochasticVehicleScheduling.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(
            StochasticVehicleScheduling;
            deps_compat=false,
            project_extras=true,
            ambiguities=false,
        )
    end

    @testset "Tasks" begin
        include("subfiles/task.jl")
    end

    @testset "City" begin
        include("subfiles/city.jl")
        include("subfiles/mini_example.jl")
    end

    @testset "Algorithms" begin
        include("subfiles/compare_algorithms.jl")
    end
end
