"""
    City

Contains all the relevant information for an instance of the stochastic VSP problem.

# Fields
- `width::Int`: city width (in minutes)
- `vehicle_cost::Float64`: cost of a vehicle in the objective function
- `delay_cost::Float64`: cost of one minute delay in the objective function
- `nb_tasks::Int`: number of tasks to fulfill
- `tasks::Vector{Task}`: tasks list (see [`Task`](@ref)), that should be ordered by start time
- `district_width::Int`: width (in minutes) of each district
- `districts::Matrix{District}`: districts matrix (see [`District`](@ref)),
    indices corresponding to their relative positions
- `random_inter_area_factor::LogNormal{Float64}`: a log-normal distribution modeling delay
    between districts
- `scenario_inter_area_factor::Matrix{Float64}`: size (nb_scenarios, 24),
    each row correspond to one scenario, each column to one hour of the day
"""
struct City
    width::Int
    # Objectives ponderation
    vehicle_cost::Float64
    delay_cost::Float64
    # Tasks
    nb_tasks::Int
    tasks::Vector{Task}
    # Stochastic specific stuff
    district_width::Int
    districts::Matrix{District}
    random_inter_area_factor::LogNormal{Float64}
    scenario_inter_area_factor::Matrix{Float64}
end

function City(;
    nb_scenarios=default_nb_scenarios,
    width=default_width,
    vehicle_cost=default_vehicle_cost,
    nb_tasks=default_nb_tasks,
    tasks=Vector{Task}(undef, nb_tasks + 2),
    district_width=default_district_width,
    districts=Matrix{District}(undef, width ÷ district_width, width ÷ district_width),
    delay_cost=default_delay_cost,
    random_inter_area_factor=default_random_inter_area_factor,
    scenario_inter_area_factor=zeros(nb_scenarios, 24),
)
    return City(
        width,
        vehicle_cost,
        delay_cost,
        nb_tasks,
        tasks,
        district_width,
        districts,
        random_inter_area_factor,
        scenario_inter_area_factor,
    )
end

"""
    create_random_city(;
        αᵥ_low=default_αᵥ_low,
        αᵥ_high=default_αᵥ_high,
        first_begin_time=default_first_begin_time,
        last_begin_time=default_last_begin_time,
        district_μ=default_district_μ,
        district_σ=default_district_σ,
        task_μ=default_task_μ,
        task_σ=default_task_σ,
        city_kwargs...
    )

- Create a city from `city_kwargs`
- Depot location at city center
- Randomize tasks, and add two dummy tasks : one `source` task at time=0 from the depot,
    and one `destination` task ending at time=end at depot
- Roll every scenario.
"""
function create_random_city(;
    αᵥ_low=default_αᵥ_low,
    αᵥ_high=default_αᵥ_high,
    first_begin_time=default_first_begin_time,
    last_begin_time=default_last_begin_time,
    district_μ=default_district_μ,
    district_σ=default_district_σ,
    task_μ=default_task_μ,
    task_σ=default_task_σ,
    city_kwargs...,
)
    city = City(; city_kwargs...)
    init_districts(city, district_μ, district_σ)
    init_tasks(city, αᵥ_low, αᵥ_high, first_begin_time, last_begin_time, task_μ, task_σ)
    generate_scenarios(city)
    compute_perturbed_end_times!(city)
    return city
end

function init_districts(city::City, district_μ::Distribution, district_σ::Distribution)
    nb_scenarios = size(city.scenario_inter_area_factor, 1)
    nb_district_per_edge = city.width ÷ city.district_width
    for x in 1:nb_district_per_edge
        for y in 1:nb_district_per_edge
            μ = rand(district_μ)
            σ = rand(district_σ)
            city.districts[x, y] = District(;
                random_delay=LogNormal(μ, σ), nb_scenarios=nb_scenarios
            )
        end
    end
    return nothing
end

function init_tasks(
    city::City,
    αᵥ_low::Real,
    αᵥ_high::Real,
    first_begin_time::Real,
    last_begin_time::Real,
    task_μ::Distribution,
    task_σ::Distribution,
)
    nb_scenarios = size(city.scenario_inter_area_factor, 1)

    point_distribution = Uniform(0, city.width)
    start_time_distribution = Uniform(first_begin_time, last_begin_time)
    travel_time_multiplier_distribution = Uniform(αᵥ_low, αᵥ_high)

    for i_task in 1:(city.nb_tasks)
        start_point = draw_random_point(point_distribution)
        end_point = draw_random_point(point_distribution)

        start_time = rand(start_time_distribution)
        end_time =
            start_time +
            rand(travel_time_multiplier_distribution) * distance(start_point, end_point)

        μ = rand(task_μ)
        σ = rand(task_σ)
        random_delay = LogNormal(μ, σ)

        city.tasks[i_task + 1] = Task(;
            type=job::TaskType,
            start_point=start_point,
            end_point=end_point,
            start_time=start_time,
            end_time=end_time,
            random_delay=random_delay,
            nb_scenarios=nb_scenarios,
        )
    end

    # add start and final "artificial" tasks
    city_center = Point(city.width / 2, city.width / 2)  # ? hard coded ?
    city.tasks[1] = Task(;
        type=depot_start::TaskType,
        start_point=city_center,
        end_point=city_center,
        start_time=0.0,
        end_time=0.0,
        random_delay=ZERO_UNIFORM,
        nb_scenarios=nb_scenarios,
    )
    final_task_time = 24 * 60.0 # ? hard coded ?
    city.tasks[end] = Task(;
        type=depot_end::TaskType,
        start_point=city_center,
        end_point=city_center,
        start_time=final_task_time,
        end_time=final_task_time,
        random_delay=ZERO_UNIFORM,
        nb_scenarios=nb_scenarios,
    )

    # sort tasks by start time
    sort!(city.tasks; by=task -> task.start_time, rev=false)
    return nothing
end

"""
    get_district(point::Point, city::City)

Return indices of the `city` district containing `point`.
"""
function get_district(point::Point, city::City)
    return trunc(Int, point.x / city.district_width) + 1,
    trunc(Int, point.y / city.district_width) + 1
end

function generate_scenarios(city::City)
    # roll all tasks
    for task in city.tasks
        roll(task)
    end

    # roll all districts
    for district in city.districts
        roll(district)
    end

    # roll inter-district
    nb_scenarios, nb_hours = size(city.scenario_inter_area_factor)
    for s in 1:nb_scenarios
        previous_delay = 0.0
        for h in 1:nb_hours
            previous_delay = (previous_delay + 0.1) * rand(city.random_inter_area_factor) # TODO : study formula
            city.scenario_inter_area_factor[s, h] = previous_delay
        end
    end
    return nothing
end

function compute_perturbed_end_times!(city::City)
    nb_scenarios = size(city.scenario_inter_area_factor, 1)

    for task in city.tasks[2:(end - 1)]
        start_time = task.start_time
        end_time = task.end_time
        start_point = task.start_point
        end_point = task.end_point

        origin_x, origin_y = get_district(start_point, city)
        destination_x, destination_y = get_district(end_point, city)
        origin_district = city.districts[origin_x, origin_y]
        destination_district = city.districts[destination_x, destination_y]

        scenario_start_time = task.scenario_start_time
        origin_delay = origin_district.scenario_delay
        destination_delay = destination_district.scenario_delay
        inter_area_delay = city.scenario_inter_area_factor

        for s in 1:nb_scenarios
            ξ₁ = scenario_start_time[s]
            ξ₂ = ξ₁ + origin_delay[s, hour_of(ξ₁)]
            ξ₃ = ξ₂ + end_time - start_time + inter_area_delay[s, hour_of(ξ₂)]
            task.scenario_end_time[s] = ξ₃ + destination_delay[s, hour_of(ξ₃)]
        end
    end
    return nothing
end

"""
    get_perturbed_travel_time(city::City, old_task_index::Int, new_task_index::Int, scenario::Int)

Compute the achieved travel time of scenario `scenario` from `old_task_index` to `new_task_index`.
"""
function get_perturbed_travel_time(
    city::City, old_task_index::Int, new_task_index::Int, scenario::Int
)
    old_task = city.tasks[old_task_index]
    new_task = city.tasks[new_task_index]

    origin_x, origin_y = get_district(old_task.end_point, city)
    destination_x, destination_y = get_district(new_task.start_point, city)

    ξ₁ = old_task.scenario_end_time[scenario]
    ξ₂ = ξ₁ + city.districts[origin_x, origin_y].scenario_delay[scenario, hour_of(ξ₁)]
    ξ₃ =
        ξ₂ +
        distance(old_task.end_point, new_task.start_point) +
        city.scenario_inter_area_factor[scenario, hour_of(ξ₂)]
    return ξ₃ + city.districts[destination_x, destination_y].scenario_delay[
        scenario, hour_of(ξ₃)
    ] - ξ₁
end

"""
    create_VSP_graph(city::City)

Return a `MetaDiGraph` computed from `city`.
Each vertex represents a task. Vertices are ordered by start time of corresponding task.
There is an edge from task u to task v the (end time of u + tie distance between u and v <= start time of v).
Every (u, v) edge has a :travel_time property, corresponding to time istance between u and v.
"""
function create_VSP_graph(city::City)
    # Initialize directed graph
    nb_vertices = city.nb_tasks + 2
    graph = SimpleDiGraph(nb_vertices)
    starting_task = 1
    end_task = nb_vertices
    job_tasks = 2:(city.nb_tasks + 1)

    travel_times = [
        distance(task1.end_point, task2.start_point) for task1 in city.tasks,
        task2 in city.tasks
    ]

    # Create existing edges
    for iorigin in job_tasks
        # link every task to base
        add_edge!(graph, starting_task, iorigin)
        add_edge!(graph, iorigin, end_task)

        for idestination in (iorigin + 1):(city.nb_tasks + 1)
            travel_time = travel_times[iorigin, idestination]
            origin_end_time = city.tasks[iorigin].end_time
            destination_begin_time = city.tasks[idestination].start_time # get_prop(graph, idestination, :task).start_time

            # there is an edge only if we can reach destination from origin before start of task
            if origin_end_time + travel_time <= destination_begin_time
                add_edge!(graph, iorigin, idestination)
            end
        end
    end

    return graph
end

"""
    compute_slacks(city, old_task_index, new_task_index)

Compute slack for features.
"""
function compute_slacks(city::City, old_task_index::Int, new_task_index::Int)
    old_task = city.tasks[old_task_index]
    new_task = city.tasks[new_task_index]

    travel_time = distance(old_task.end_point, new_task.start_point)
    perturbed_end_times = old_task.scenario_end_time
    perturbed_start_times = new_task.scenario_start_time

    return perturbed_start_times .- (perturbed_end_times .+ travel_time)
end

"""
    compute_features(city::City)

Returns a matrix of features of size (20, nb_edges).
For each edge, compute the following features (in the same order):
- travel time
- vehicle_cost if edge is connected to source, else 0
- 9 deciles of the slack
- cumulative probability distribution of the slack evaluated in [-100, -50, -20, -10, 0, 10, 50, 200, 500]
"""
function compute_features(city::City)
    graph = create_VSP_graph(city)

    cumul = [-100, -50, -20, -10, 0, 10, 50, 200, 500]
    nb_features = 2 + 9 + length(cumul)
    features = zeros(nb_features, ne(graph))

    # features indices
    travel_time_index = 1
    connected_to_source_index = 2
    slack_deciles_indices = 3:11
    slack_cumulative_distribution_indices = 12:nb_features

    for (i, edge) in enumerate(edges(graph))
        # compute travel time
        features[travel_time_index, i] = distance(
            city.tasks[src(edge)].end_point, city.tasks[dst(edge)].start_point
        )
        # if edge connected to source node
        features[connected_to_source_index, i] = src(edge) == 1 ? city.vehicle_cost : 0.0

        # slack related features
        slacks = compute_slacks(city, src(edge), dst(edge))
        # compute deciles
        features[slack_deciles_indices, i] = quantile(slacks, [0.1 * i for i in 1:9])
        # compute cumulative distribution
        features[slack_cumulative_distribution_indices, i] = [
            mean(slacks .<= x) for x in cumul
        ]
    end
    return features
end

"""
    compute_slacks(city)

Compute slack for instance.
TODO: differentiate from other method
"""
function compute_slacks(city::City, graph::AbstractGraph)
    (; tasks) = city
    N = nv(graph)
    slack_list = [
        [
            (dst(e) < N ? tasks[dst(e)].scenario_start_time[ω] : Inf) -
            (tasks[src(e)].end_time + get_perturbed_travel_time(city, src(e), dst(e), ω))
            for ω in 1:get_nb_scenarios(city)
        ] for e in edges(graph)
    ]
    I = [src(e) for e in edges(graph)]
    J = [dst(e) for e in edges(graph)]
    return sparse(I, J, slack_list)
end

"""
    compute_delays(city)

Compute delays for instance.
"""
function compute_delays(city::City)
    nb_tasks = get_nb_tasks(city)
    nb_scenarios = get_nb_scenarios(city)
    ε = zeros(nb_tasks, nb_scenarios)
    for (index, task) in enumerate(city.tasks)
        ε[index, :] .= task.scenario_end_time .- task.end_time
    end
    return ε
end

"""
    get_nb_tasks(city::City)

Returns the number of tasks in city.
"""
function get_nb_tasks(city::City)
    return length(city.tasks)
end

"""
    get_nb_scenarios(city::City)

Returns the number of scenarios in city.
"""
function get_nb_scenarios(city::City)
    return size(city.scenario_inter_area_factor, 1)
end
