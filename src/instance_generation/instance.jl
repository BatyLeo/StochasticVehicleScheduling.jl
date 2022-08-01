"""
    Instance{G<:AbstractGraph,M1,M2}

Instance of the stochastic VSP problem.

# Fields
- `city::City`
- `graph::G`: graph computed from `city` with the `create_VSP_graph(city::City)` method.
- `features::Matrix{Float64}`: features matrix computed from `city`
- `slacks`
- `delays`
"""
struct Instance{G<:AbstractGraph,M1<:AbstractMatrix,M2<:AbstractMatrix}
    city::City
    graph::G
    features::Matrix{Float64}
    slacks::M1
    delays::M2
end

"""
    Instance(city::City)

Build an `Instance` from a `City`, by computing its graph, features, slacks and delays.
"""
function Instance(city::City)
    graph = create_VSP_graph(city)
    features = compute_features(city)
    slacks = compute_slacks(city, graph)
    delays = compute_delays(city)
    return Instance(city, graph, features, slacks, delays)
end

"""
    get_nb_scenarios(instance)

Returns the number of scenarios in instance.
"""
get_nb_scenarios(instance::Instance) = size(instance.city.scenario_inter_area_factor, 1)

"""
    get_nb_tasks(instance)

Returns the number of tasks in instance.
"""
get_nb_tasks(instance::Instance) = nv(instance.graph) - 2

"""
    create_random_instance([; city_kwargs])

Returns a random instance created with city_kwargs.
"""
function create_random_instance(; city_kwargs...)
    return Instance(create_random_city(; city_kwargs...))
end
