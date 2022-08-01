using Distributions
import StochasticVehicleScheduling: Point, roll

start_point = Point(0, 0)
end_point = Point(5, 5)
start_time = 0.0
end_time = 10.0
nb_scenarios = 5

@testset "Deterministic task" begin
    # Create a task with default random_delay, i.e. always equal to 0
    deterministic_task = StochasticVehicleScheduling.Task(;
        start_point=start_point,
        end_point=end_point,
        start_time=start_time,
        end_time=end_time,
        nb_scenarios=nb_scenarios,
    )
    # scenario_start_time and scenario_end_time arrays should have been respectively
    # initialized with start_time and end_time
    @test all(deterministic_task.scenario_start_time .== start_time)
    @test all(deterministic_task.scenario_end_time .== end_time)

    # Draw scenarios
    roll(deterministic_task)

    # scenario arrays should not have changed
    @test all(deterministic_task.scenario_start_time .== start_time)
    @test all(deterministic_task.scenario_end_time .== end_time)
end

@testset "Stochastic task" begin
    # Create a task with default a log normal random_delay
    random_delay = LogNormal(2.0, 1.0)
    stochastic_task = StochasticVehicleScheduling.Task(;
        start_point=start_point,
        end_point=end_point,
        start_time=start_time,
        end_time=end_time,
        nb_scenarios=nb_scenarios,
        random_delay=random_delay,
    )
    # scenario_start_time and scenario_end_time arrays should have been respectively
    # initialized with start_time and end_time
    @test all(stochastic_task.scenario_start_time .== start_time)
    @test all(stochastic_task.scenario_end_time .== end_time)

    # Draw scenarios
    roll(stochastic_task)

    # This time, scenario_start_time should have been increased,
    # and scenrio_end_time be unchanged
    @test all(stochastic_task.scenario_start_time .>= start_time)
    @test all(stochastic_task.scenario_end_time .== end_time)
end
