using Aqua
using JuliaFormatter
using StochasticVehicleScheduling
using Test

include("subfiles/utils.jl")

@testset verbose = true "StochasticVehicleScheduling.jl" begin
    @testset "Code" begin
        include("subfiles/code.jl")
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
