function solve_scenarios(instance::Instance; model_builder=cbc_model)
    (; graph, slacks, delays, city) = instance
    (; delay_cost, vehicle_cost) = city
    nb_nodes = nv(graph)
    job_indices = 2:nb_nodes-1
    nodes = 1:nb_nodes

    # Pre-processing
    ε = delays
    #Rmax = maximum(ε, dims=1)
    Rmax = maximum(sum(ε, dims=1))
    nb_scenarios = size(ε, 2)
    Ω = 1:nb_scenarios

    # Model definition
    model = model_builder()

    # Variables and objective function
    @variable(model, y[u in nodes, v in nodes; has_edge(graph, u, v)], Bin)
    @variable(model, R[v in nodes, ω in Ω] >= 0) # propagated delay of job v
    @variable(model, yR[u in nodes, v in nodes, ω in Ω; has_edge(graph, u, v)] >= 0) # yR[u, v] = y[u, v] * R[u, ω]

    @objective(
        model,
        Min,
        delay_cost * sum(sum(R[v, ω] for v in job_indices) for ω in Ω) / nb_scenarios # average total delay
            + vehicle_cost * sum(y[1, v] for v in job_indices) # nb_vehicles
    )

    # Flow contraints
    @constraint(
        model,
        flow[i in job_indices],
        sum(y[j, i] for j in inneighbors(graph, i)) ==
        sum(y[i, j] for j in outneighbors(graph, i))
    )
    @constraint(
        model,
        unit_demand[i in job_indices],
        sum(y[j, i] for j in inneighbors(graph, i)) == 1
    )

    # Delay propagation constraints
    @constraint(model, [ω in Ω], R[1, ω] == ε[1, ω])
    @constraint(model, R_delay_1[v in job_indices, ω in Ω], R[v, ω] >= ε[v, ω])
    @constraint(
        model,
        R_delay_2[v in job_indices, ω in Ω],
        R[v, ω] >= ε[v, ω] + sum(yR[u, v, ω] - y[u, v] * slacks[u, v][ω] for u in nodes if has_edge(graph, u, v))
    )

    # Mc Cormick linearization constraints
    @constraint(
        model,
        R_McCormick_1[u in nodes, v in nodes, ω in Ω; has_edge(graph, u, v)],
        yR[u, v, ω] >= R[u, ω] + Rmax * (y[u, v] - 1)
    )
    @constraint(
        model,
        R_McCormick_2[u in nodes, v in nodes, ω in Ω; has_edge(graph, u, v)],
        yR[u, v, ω] <= Rmax * y[u, v]
    )

    # Solve model
    optimize!(model)
    solution = value.(y)

    paths = Vector{Int}[]
    for i in job_indices
        if solution[1, i] ≈ 1
            new_path = [1, i]
            index = i
            while index < nb_nodes
                for j in outneighbors(graph, index)
                    if solution[index, j] ≈ 1
                        push!(new_path, j)
                        index = j
                        break
                    end
                end
            end
            push!(paths, new_path)
        end
    end

    return objective_value(model), paths
end
