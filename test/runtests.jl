using Aqua
using JuliaFormatter
using StochasticVehicleScheduling
using Test

include("subfiles/utils.jl")

format(StochasticVehicleScheduling; verbose=true)

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
