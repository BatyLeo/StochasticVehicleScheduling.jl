"""
    easy_problem(θ[; instance, model_builder])

Solves the easy problem of the learning pipeline given arcs weights θ.
Note: If you have Gurobi, use `grb_model` as `model_builder` instead od `cbc_model`.
"""
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
        if isapprox(value(y[edge.src, edge.dst]), 1; atol=1e-3)
            solution[i] = true
        end
    end

    return solution
end
