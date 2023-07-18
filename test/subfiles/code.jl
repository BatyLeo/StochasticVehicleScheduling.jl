@testset "Quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(StochasticVehicleScheduling; ambiguities=false)
end

@testset "Correctness (JET.jl)" begin
    using JET
    using JLD2
    using NamedTupleTools
    using Core.Compiler
    if VERSION >= v"1.8"
        JET.test_package(
            StochasticVehicleScheduling;
            toplevel_logger=nothing,
            ignored_modules=(JLD2, Base, NamedTupleTools, Core.Compiler),
            mode=:typo,
        )
    end
end

@testset "Formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    @test format(StochasticVehicleScheduling; verbose=false, overwrite=false)
end
