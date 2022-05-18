@enum TaskType depot_start job depot_end

"""
    Task

# Fields
- `type::TaskType`
- `start_point::Point`: starting location of the task
- `end_point::Point`: end location of the task
- `start_time::Float64`: start time (in minutes) of the task
- `end_time::Float64`: end time (in minutes) of the task
- `random_delay::LogNormal{Float64}`: lognormal distribution modeling the task start delay
- `scenario_start_time::Vector{Float64}`: size (nb_scenarios),
    realized delayed start times for each scenario
- `scenario_end_time::Vector{Float64}`: size (nb_scenarios),
    realized delayed end times for each scenario
"""
struct Task
    type::TaskType
    start_point::Point
    end_point::Point
    start_time::Float64
    end_time::Float64
    random_delay::LogNormal{Float64}
    scenario_start_time::Vector{Float64}
    scenario_end_time::Vector{Float64}
end

function Task(;
    type::TaskType,
    start_point::Point,
    end_point::Point,
    start_time::Float64,
    end_time::Float64,
    nb_scenarios::Int,
    random_delay::LogNormal{Float64}=ZERO_UNIFORM,
)
    return Task(type, start_point, end_point, start_time, end_time,
        random_delay, zeros(nb_scenarios), zeros(nb_scenarios))
end

"""
    roll(task)

Populate `scenario_start_time` with delays drawn from `random_delay` distribution
for each scenario.
"""
function roll(task::Task)
    nb_scenarios = length(task.scenario_start_time)
    task.scenario_start_time .= task.start_time .+ rand(task.random_delay, nb_scenarios)
    # # Draw each scenario
    # for s in eachindex(task.scenario_start_time)
    #     delay = rand(task.random_delay)
    #     task.scenario_start_time[s] = task.start_time + delay
    # end
    return nothing
end

Base.show(io::IO, task::Task) = @printf(
    "(%.2f, %.2f) -> (%.2f, %.2f), [%.2f, %.2f], %s, %s, %s",
    task.start_point.x,
    task.start_point.y,
    task.end_point.x,
    task.end_point.y,
    task.start_time,
    task.end_time,
    task.type,
    task.scenario_start_time,
    task.scenario_end_time,
)
