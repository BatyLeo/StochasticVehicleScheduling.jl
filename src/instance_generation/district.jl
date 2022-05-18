"""
    District

# Fields
- `random_delay::LogNormal{Float64}`: log-normal distribution modeling the district delay
- `scenario_delay::Matrix{Float64}`: size (nb_scenarios, 24),
    observed delays for each scenario and hour of the day
"""
struct District
    random_delay::LogNormal{Float64}
    scenario_delay::Matrix{Float64}
end

function District(; random_delay::LogNormal{Float64}, nb_scenarios::Int)
    return District(random_delay, zeros(nb_scenarios, 24))
end

function scenario_next_delay(previous_delay::Real, random_delay::Distribution)
    return previous_delay / 2.0 + rand(random_delay)
end

"""
    roll(task)

Populate `scenario_delay` with delays drawn from `random_delay` distribution
for each (scenario, hour) pair.
"""
function roll(district::District)
    nb_scenarios, nb_hours = size(district.scenario_delay)
    # Loop on scenarios
    for s in 1:nb_scenarios
        previous_delay = 0.0
        # Loop on hours
        for h in 1:nb_hours
            previous_delay = scenario_next_delay(previous_delay, district.random_delay)
            district.scenario_delay[s, h] = previous_delay
        end
    end
    return nothing
end
