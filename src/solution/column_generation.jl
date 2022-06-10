function delay_sum(path, slacks, delays)
    nb_scenarios = size(delays, 2)
    old_v = path[1]
    R = delays[old_v, :]
    C = 0.0
    for v in path[2:end-1]
        @. R = max(R - slacks[old_v, v], 0) + delays[v, :]
        C += sum(R) / nb_scenarios
        old_v = v
    end
    return C
end

function column_generation(instance::Instance)
    (; graph, slacks, delays, city) = instance

    vehicle_cost = city.vehicle_cost
    delay_cost = city.delay_cost

    nb_nodes = nv(graph)
    job_indices = 2:nb_nodes-1

    model = Model(GLPK.Optimizer)

    @variable(model, λ[v in 1:nb_nodes])

    @objective(model, Max, sum(λ[v] for v in job_indices))

    initial_paths = [[1, v, nb_nodes] for v in job_indices]
    @constraint(
        model,
        con[p in initial_paths],
        delay_cost * delay_sum(p, slacks, delays) + vehicle_cost
            - sum(λ[v] for v in job_indices if v in p) >= 0
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
        a = delay_cost * delay_sum(p_star, slacks, delays) + vehicle_cost - sum(λ_val[v] for v in job_indices if v in p_star)
        #@info "Hello" delay_sum(p_star, slacks, delays) c_star
        #@info "adding" c_star p_star path_cost a path_cost - λ_sum
        if path_cost - λ_sum > -1e-10
            break
        end
        push!(new_paths, p_star)
        push!(cons, @constraint(
            model,
            path_cost - sum(λ[v] for v in job_indices if v in p_star) >= 0
        ))
    end

    return value.(λ), objective_value(model), cat(initial_paths, new_paths; dims=1), dual.(con), dual.(cons)
end

function compute_solution_from_selected_columns(instance::Instance, paths; bin=true)
    (; graph, slacks, delays, city) = instance
    (; vehicle_cost, delay_cost) = city

    nb_nodes = nv(graph)
    job_indices = 2:nb_nodes-1

    model = Model(GLPK.Optimizer)

    if bin
        @variable(model, y[p in paths], Bin)
    else
        @variable(model, y[p in paths] >= 0)
    end

    @objective(model, Min,
        sum((delay_cost * delay_sum(p, slacks, delays) + vehicle_cost) * y[p] for p in paths)
    )

    @constraint(
        model,
        con[v in job_indices],
        sum(y[p] for p in paths if v in p) == 1
    )

    optimize!(model)

    return objective_value(model), value.(y)
end