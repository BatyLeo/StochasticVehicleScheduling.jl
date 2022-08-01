"""
    Instance{G<:AbstractGraph}

Instance of the stochastic VSP problem.

# Fields
- `city::City`
- `graph::G`: graph computed from `city` with the `create_VSP_graph(city::City)` method.
- `features::Matrix{Float64}`: features matrix computed from `city`
"""
struct Instance{G<:AbstractGraph,M1<:AbstractMatrix,M2<:AbstractMatrix}
    city::City
    graph::G
    features::Matrix{Float64}
    slacks::M1
    delays::M2
end

function Instance(city::City)
    graph = create_VSP_graph(city)
    features = compute_features(city)
    slacks = compute_slacks(city, graph)
    delays = compute_delays(city)
    return Instance(city, graph, features, slacks, delays)
end

get_nb_scenarios(instance::Instance) = size(instance.city.scenario_inter_area_factor, 1)

function get_nb_tasks(instance::Instance)
    return nv(instance.graph) - 2
end
