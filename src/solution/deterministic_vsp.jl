function cbc_model()
    model = Model(Cbc.Optimizer)
    set_optimizer_attribute(model, "logLevel", 0)
    return model
end

function glpk_model()
    model = Model(GLPK.Optimizer)
    return model
end

"""
    solve_deterministic_VSP(instance::Instance; include_delays=true)

Return the optimal solution of the deterministic VSP problem associated to `instance`.
The objective function is `vehicle_cost * nb_vehicles + include_delays * delay_cost * sum_of_travel_times`
"""
function solve_deterministic_VSP(instance::Instance; include_delays=true, model_builder=cbc_model)
    (; city, graph) = instance

    travel_times = [
        distance(task1.end_point, task2.start_point) for task1 in city.tasks,
        task2 in city.tasks
    ]

    model = model_builder()
    # set_optimizer_attribute(model, "logLevel", 0)
    #model = Model(gurobi_optimizer)
    # set_optimizer_attribute(model, "OutputFlag", 0)

    nb_nodes = nv(graph)
    job_indices = 2:nb_nodes-1

    # @variable(model, x[i=1:nb_nodes, j=1:nb_nodes; has_edge(graph, i, j)], Bin)
    @variable(model, x[i=1:nb_nodes, j=1:nb_nodes; has_edge(graph, i, j)], Bin)

    @objective(
        model,
        Min,
        instance.city.vehicle_cost * sum(x[1, j] for j in job_indices) +
        include_delays * instance.city.delay_cost * sum(
            travel_times[i, j] * x[i, j] for i in 1:nb_nodes for
            j in 1:nb_nodes if has_edge(graph, i, j)
        )
    )

    @constraint(
        model,
        flow[i in job_indices],
        sum(x[j, i] for j in inneighbors(graph, i)) ==
        sum(x[i, j] for j in outneighbors(graph, i))
    )
    @constraint(
        model,
        demand[i in job_indices],
        sum(x[j, i] for j in inneighbors(graph, i)) == 1
    )

    optimize!(model)

    solution = solution_from_JuMP_array(value.(x), graph)

    return objective_value(model), solution
end

function easy_problem(θ::AbstractVector; instance::Instance, model_builder=cbc_model)
    (; graph) = instance

    model = model_builder()

    nb_nodes = nv(graph)
    job_indices = 2:nb_nodes-1

    @variable(model, y[i=1:nb_nodes, j=1:nb_nodes; has_edge(graph, i, j)], Bin)

    @objective(model, Max, sum(θ[i] * y[edge.src, edge.dst] for (i, edge) in enumerate(edges(graph))))

    @constraint(
        model,
        flow[i in job_indices],
        sum(y[j, i] for j in inneighbors(graph, i)) ==
        sum(y[i, j] for j in outneighbors(graph, i))
    )
    @constraint(
        model,
        demand[i in job_indices],
        sum(y[j, i] for j in inneighbors(graph, i)) == 1
    )

    optimize!(model)

    solution = falses(ne(graph))
    for (i, edge) in enumerate(edges(graph))
        if value(y[edge.src, edge.dst]) ≈ 1
            solution[i] = true
        end
    end

    return solution
end

# function solve_scenario(instance::Instance; scenario_index::Int=1, model_builder=cbc_model)
#     graph, city = instance.graph, instance.city

#     nb_nodes = nv(graph)
#     job_indices = 2:nb_nodes-1
#     nodes = 1:nb_nodes

#     # Pre-processing
#     ε = [task.scenario_end_time[scenario_index] for task in city.tasks]
#     Rmax = sum(ε)
#     slack = zeros(nb_nodes, nb_nodes)
#     for edge in edges(graph)
#         u, v = edge.src, edge.dst
#         slack[u, v] = city.tasks[v].scenario_start_time[scenario_index] - get_perturbed_travel_time(city, u, v, scenario_index)
#     end

#     # Model definition
#     model = model_builder()
#     # set_optimizer_attribute(model, "logLevel", 0)
#     # model = Model(gurobi_optimizer)
#     # set_optimizer_attribute(model, "OutputFlag", 0)

#     # Variables and objective function
#     @variable(model, x[u in nodes, v in nodes; has_edge(graph, u, v)], Bin)
#     @variable(model, τ[v in job_indices] >= 0) # end time of job v
#     @variable(model, Δ[v in job_indices] >= 0) # propagated delay of job v
#     @variable(model, zτ[u in job_indices, v in nodes; has_edge(graph, u, v)] >= 0) # zτ[u, v] = x[u, v] * τ[u]
#     @variable(model, zΔ[u in job_indices, v in nodes; has_edge(graph, u, v)] >= 0) # zΔ[u, v] = x[u, v] * Δ[u]

#     @objective(
#         model,
#         Min,
#         city.vehicle_cost * sum(x[1, j] for j in job_indices) +
#         city.delay_cost * sum(Δ[v] for v in job_indices) # total delay
#     )

#     # Flow contraints
#     @constraint(
#         model,
#         flow[i in job_indices],
#         sum(x[j, i] for j in inneighbors(graph, i)) ==
#         sum(x[i, j] for j in outneighbors(graph, i))
#     )
#     @constraint(
#         model,
#         unit_demand[i in job_indices],
#         sum(x[j, i] for j in inneighbors(graph, i)) == 1
#     )

#     # Delay propagation constraints
#     @constraint(model, τ_delay_1[v in job_indices], τ[v] >= ε[v])
#     @constraint(
#         model,
#         τ_delay_2[v in job_indices],
#         τ[v] >= ε[v] + sum(zτ[u, v] - x[u, v] * slack[u, v] for u in job_indices if has_edge(graph, u, v))
#     )

#     @constraint(model, Δ_delay_3[v in job_indices], Δ[v] >= sum(zΔ[u, v] for u in job_indices if has_edge(graph, u, v)))
#     @constraint(
#         model,
#         Δ_delay_4[v in job_indices],
#         Δ[v] >= sum(zΔ[u, v] + zτ[u, v] - x[u, v] * slack[u, v] for u in job_indices if has_edge(graph, u, v))
#     )

#     # Mc Cormick linearization constraints
#     @constraint(
#         model,
#         τ_McCormick_1[u in job_indices, v in nodes; has_edge(graph, u, v)],
#         zτ[u, v] >= τ[u] + Rmax * (x[u, v] - 1)
#     )
#     @constraint(
#         model,
#         τ_McCormick_2[u in job_indices, v in nodes; has_edge(graph, u, v)],
#         zτ[u, v] <= Rmax * x[u, v]
#     )

#     @constraint(
#         model,
#         Δ_McCormick_1[u in job_indices, v in nodes; has_edge(graph, u, v)],
#         zΔ[u, v] >= Δ[u] + Rmax * (x[u, v] - 1)
#     )
#     @constraint(
#         model,
#         Δ_McCormick_2[u in job_indices, v in nodes; has_edge(graph, u, v)],
#         zΔ[u, v] <= Rmax * x[u, v]
#     )

#     # Solve model
#     optimize!(model)
#     solution = solution_from_JuMP_array(value.(x), graph)

#     # @info "Delays" value.(Δ)

#     return objective_value(model), solution
# end

# function solve_scenario(instance::Instance; scenario_index::Int=1)
#     graph = instance.graph
#     nb_nodes = nv(graph)
#     city = instance.city

#     ε = [task.scenario_end_time[scenario_index] - task.end_time for task in city.tasks]
#     Rmax = sum(ε)
#     slack = zeros(nb_nodes, nb_nodes)
#     for edge in edges(graph)
#         u, v = edge.src, edge.dst
#         slack[u, v] = -(city.tasks[u].end_time - city.tasks[v].scenario_start_time[scenario_index] + get_perturbed_travel_time(city, u, v, scenario_index))
#     end

#     model = Model(gurobi_optimizer)
#     set_optimizer_attribute(model, "OutputFlag", 0)
#     #set_optimizer_attribute(model, "msg_lev", GLPK.GLP_MSG_ALL)

#     job_indices = 2:nb_nodes-1
#     nodes = 1:nb_nodes

#     @variable(model, x[i in nodes, j in nodes; has_edge(graph, i, j)], Bin)
#     @variable(model, r[v in job_indices] >= 0)
#     # z_uv = x_uv * r_u
#     @variable(model, z[u in job_indices, v in job_indices; has_edge(graph, u, v)] >= 0)

#     @objective(
#         model,
#         Min,
#         city.vehicle_cost * sum(x[1, j] for j in job_indices) +
#         city.delay_cost * sum(r[v] for v in job_indices)
#     )

#     @constraint(
#         model,
#         flow[i in job_indices],
#         sum(x[j, i] for j in inneighbors(graph, i)) ==
#         sum(x[i, j] for j in outneighbors(graph, i))
#     )
#     @constraint(
#         model,
#         demand[i in job_indices],
#         sum(x[j, i] for j in inneighbors(graph, i)) == 1
#     )

#     @constraint(model, delay2[v in job_indices], r[v] >= ε[v])
#     @constraint(
#         model,
#         delay[v in job_indices],
#         r[v] >= ε[v] + sum(z[u, v] - x[u, v] * slack[u, v] for u in job_indices if has_edge(graph, u, v))
#     )

#     @constraint(
#         model,
#         McCormick1[u in job_indices, v in job_indices; has_edge(graph, u, v)],
#         z[u, v] >= r[u] + Rmax * (x[u, v] - 1)
#     )

#     @constraint(
#         model,
#         McCormick2[u in job_indices, v in job_indices; has_edge(graph, u, v)],
#         z[u, v] <= Rmax * x[u, v]
#     )

#     optimize!(model)

#     solution = solution_from_JuMP_array(value.(x), graph)

#     return objective_value(model), solution
# end

# function solve_scenario2(instance::Instance; scenario_index::Int=1)
#     graph = instance.graph
#     nb_nodes = nv(graph)
#     city = instance.city

#     ε = [task.scenario_end_time[scenario_index] for task in city.tasks]
#     Rmax = sum(ε)
#     slack = zeros(nb_nodes, nb_nodes)
#     for edge in edges(graph)
#         u, v = edge.src, edge.dst
#         slack[u, v] = city.tasks[v].scenario_start_time[scenario_index] - get_perturbed_travel_time(city, u, v, scenario_index)
#     end

#     model = Model(gurobi_optimizer)
#     set_optimizer_attribute(model, "OutputFlag", 0)

#     job_indices = 2:nb_nodes-1
#     nodes = 1:nb_nodes

#     @variable(model, x[i in nodes, j in nodes; has_edge(graph, i, j)], Bin)
#     @variable(model, r[v in job_indices] >= 0)
#     # z_uv = x_uv * r_u
#     @variable(model, z[u in job_indices, v in job_indices; has_edge(graph, u, v)] >= 0)

#     @objective(
#         model,
#         Min,
#         city.vehicle_cost * sum(x[1, j] for j in job_indices) +
#         city.delay_cost * sum(r[v] for v in job_indices)
#     )

#     @constraint(
#         model,
#         flow[i in job_indices],
#         sum(x[j, i] for j in inneighbors(graph, i)) ==
#         sum(x[i, j] for j in outneighbors(graph, i))
#     )
#     @constraint(
#         model,
#         demand[i in job_indices],
#         sum(x[j, i] for j in inneighbors(graph, i)) == 1
#     )

#     @constraint(model, delay2[v in job_indices], r[v] >= ε[v])
#     @constraint(
#         model,
#         delay[v in job_indices],
#         r[v] >= ε[v] + sum(z[u, v] - x[u, v] * slack[u, v] for u in job_indices if has_edge(graph, u, v))
#     )

#     @constraint(
#         model,
#         McCormick1[u in job_indices, v in job_indices; has_edge(graph, u, v)],
#         z[u, v] >= r[u] + Rmax * (x[u, v] - 1)
#     )

#     @constraint(
#         model,
#         McCormick2[u in job_indices, v in job_indices; has_edge(graph, u, v)],
#         z[u, v] <= Rmax * x[u, v]
#     )

#     optimize!(model)

#     solution = solution_from_JuMP_array(value.(x), graph)

#     return objective_value(model), solution
# end
