"""
    delay_sum(path, slacks, delays)

Evaluate the total delay along path.
"""
function delay_sum(path, slacks, delays)
    nb_scenarios = size(delays, 2)
    old_v = path[1]
    R = delays[old_v, :]
    C = 0.0
    for v in path[2:(end - 1)]
        @. R = max(R - slacks[old_v, v], 0) + delays[v, :]
        C += sum(R) / nb_scenarios
        old_v = v
    end
    return C
end

"""
    column_generation(instance::Instance)

Note: If you have Gurobi, use `grb_model` as `model_builder` instead of `glpk_model`.
"""
function column_generation(
    instance::AbstractInstance; only_relaxation=false, model_builder=glpk_model
)
    (; graph, slacks, delays, vehicle_cost, delay_cost) = instance

    nb_nodes = nv(graph)
    job_indices = 2:(nb_nodes - 1)

    model = model_builder()

    @variable(model, λ[v in 1:nb_nodes])

    @objective(model, Max, sum(λ[v] for v in job_indices))

    initial_paths = [[1, v, nb_nodes] for v in job_indices]
    @constraint(
        model,
        con[p in initial_paths],
        delay_cost * delay_sum(p, slacks, delays) + vehicle_cost -
        sum(λ[v] for v in job_indices if v in p) >= 0
    )
    @constraint(model, λ[1] == 0)
    @constraint(model, λ[nb_nodes] == 0)

    new_paths = Vector{Int}[]
    cons = []

    while true
        optimize!(model)
        λ_val = value.(λ)
        (; c_star, p_star) = stochastic_routing_shortest_path(
            graph, slacks, delays, λ_val ./ delay_cost
        )
        λ_sum = sum(λ_val[v] for v in job_indices if v in p_star)
        path_cost = delay_cost * c_star + λ_sum + vehicle_cost
        if path_cost - λ_sum > -1e-10
            break
        end
        push!(new_paths, p_star)
        push!(
            cons,
            @constraint(
                model, path_cost - sum(λ[v] for v in job_indices if v in p_star) >= 0
            )
        )
    end

    c_low = objective_value(model)
    paths = cat(initial_paths, new_paths; dims=1)
    c_upp, y, _ = compute_solution_from_selected_columns(instance, paths)

    # If relaxation requested or solution is optimal, return
    if c_upp ≈ c_low || only_relaxation
        return value.(λ),
        objective_value(model), cat(initial_paths, new_paths; dims=1), dual.(con),
        dual.(cons)
    end

    # else, try to close the gap
    threshold = (c_upp - c_low - vehicle_cost) / delay_cost
    λ_val = value.(λ)
    additional_paths, costs = ConstrainedShortestPaths.stochastic_routing_shortest_path_with_threshold(
        graph, slacks, delays, λ_val ./ delay_cost; threshold
    )

    return value.(λ),
    objective_value(model),
    unique(cat(initial_paths, new_paths, additional_paths; dims=1)),
    dual.(con),
    dual.(cons)
end

"""
    compute_solution_from_selected_columns(instance::AbstractInstance, paths[; bin=true])

Note: If you have Gurobi, use `grb_model` as `model_builder` instead od `glpk_model`.
"""
function compute_solution_from_selected_columns(
    instance::AbstractInstance, paths; bin=true, model_builder=glpk_model
)
    (; graph, slacks, delays, vehicle_cost, delay_cost) = instance

    nb_nodes = nv(graph)
    job_indices = 2:(nb_nodes - 1)

    model = model_builder()

    if bin
        @variable(model, y[p in paths], Bin)
    else
        @variable(model, y[p in paths] >= 0)
    end

    @objective(
        model,
        Min,
        sum(
            (delay_cost * delay_sum(p, slacks, delays) + vehicle_cost) * y[p] for p in paths
        )
    )

    @constraint(model, con[v in job_indices], sum(y[p] for p in paths if v in p) == 1)

    optimize!(model)

    sol = value.(y)
    return objective_value(model), sol, paths[[sol[p] for p in paths] .== 1.0]
end

function column_generation_algorithm(instance::AbstractInstance)
    _, _, columns, _, _ = column_generation(instance)
    _, _, sol = compute_solution_from_selected_columns(instance, columns)
    col_solution = solution_from_paths(sol, instance)
    return col_solution
end
