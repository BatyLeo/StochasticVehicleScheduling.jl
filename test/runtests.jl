using StochasticVehicleScheduling
using Random
using Test
using UnicodePlots
using Gurobi

const GRB_ENV = Gurobi.Env()

function short(solution::Solution)
    res = Vector{Int}[]
    n, m = size(solution.path_value)
    for row in 1:n
        hello = Int[]
        for col in 1:m
            if solution.path_value[row, col]
                push!(hello, col)
            end
        end
        if length(hello) > 0
            push!(res, hello)
        end
    end
    return "$(length(res)), $res"
end

@testset verbose=true "StochasticVehicleScheduling.jl" begin
    @testset "Miscellaneous" begin
        include("miscellaneous.jl")
        include("mini_example.jl")
    end

    @testset "City" begin
        include("city.jl")
    end

    @testset "Features" begin
        include("dataset.jl")
    end

    @testset "Tutorial" begin
        include("tutorial.jl")
    end
end
