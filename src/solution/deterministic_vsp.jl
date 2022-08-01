function cbc_model()
    model = Model(Cbc.Optimizer)
    set_optimizer_attribute(model, "logLevel", 0)
    return model
end

function glpk_model()
    model = Model(GLPK.Optimizer)
    return model
end

function grb_model()
    model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    set_optimizer_attribute(model, "OutputFlag", 0)
    return model
end

"""
    solve_deterministic_VSP(instance::Instance; include_delays=true)

Return the optimal solution of the deterministic VSP problem associated to `instance`.
The objective function is `vehicle_cost * nb_vehicles + include_delays * delay_cost * sum_of_travel_times`
"""
function solve_deterministic_VSP(
    instance::Instance; include_delays=true, model_builder=cbc_model
)
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
    job_indices = 2:(nb_nodes - 1)

    # @variable(model, x[i=1:nb_nodes, j=1:nb_nodes; has_edge(graph, i, j)], Bin)
    @variable(model, x[i=1:nb_nodes, j=1:nb_nodes; has_edge(graph, i, j)], Bin)

    @objective(
        model,
        Min,
        instance.city.vehicle_cost * sum(x[1, j] for j in job_indices) +
            include_delays *
        instance.city.delay_cost *
        sum(
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
        model, demand[i in job_indices], sum(x[j, i] for j in inneighbors(graph, i)) == 1
    )

    optimize!(model)

    solution = solution_from_JuMP_array(value.(x), graph)

    return objective_value(model), solution
end

function easy_problem(θ::AbstractVector; instance::Instance, model_builder=cbc_model)
    (; graph) = instance

    model = model_builder()

    nb_nodes = nv(graph)
    job_indices = 2:(nb_nodes - 1)

    @variable(model, y[i=1:nb_nodes, j=1:nb_nodes; has_edge(graph, i, j)], Bin)

    @objective(
        model,
        Max,
        sum(θ[i] * y[edge.src, edge.dst] for (i, edge) in enumerate(edges(graph)))
    )

    @constraint(
        model,
        flow[i in job_indices],
        sum(y[j, i] for j in inneighbors(graph, i)) ==
            sum(y[i, j] for j in outneighbors(graph, i))
    )
    @constraint(
        model, demand[i in job_indices], sum(y[j, i] for j in inneighbors(graph, i)) == 1
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
