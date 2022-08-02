# using BenchmarkTools
# using ConstrainedShortestPaths
# using Graphs
using Gurobi
using InferOpt
using Random
using StochasticVehicleScheduling

# dataset_path = "data/data.jld2"
# data = load(dataset_path)["data"];

# instance = data.train.X[1];
# (; city, features, graph, slacks, delays) = instance;

# slacks[1, 2]

# delays

# features

# @benchmark stochastic_routing_shortest_path(graph, slacks, delays)

# function f()
#     for _ in 1:1000
#         stochastic_routing_shortest_path(graph, slacks, delays)
#     end
#     return nothing
# end

# f()

Random.seed!(1)
instance = create_random_instance(; nb_tasks=20, nb_scenarios=10);

optimal_value, optimal_solution = solve_scenarios(instance; model_builder=grb_model);
optimal_value

# fig = plot_instance(instance);
# display(fig)
# #@profview easy_problem(ones(ne(instance.graph)); instance);

# solution = heuristic_solution(instance; nb_it=10000)
# fig2 = plot_solution(solution, instance);
# display(fig2)

#instance.delays

λ, c_low, paths, dual1, dual2 = column_generation(
    instance; only_relaxation=true, model_builder=grb_model
);
obj2, y = compute_solution_from_selected_columns(instance, paths; bin=true);
obj2

λ, c_low, paths, dual1, dual2 = column_generation(
    instance; only_relaxation=false, model_builder=grb_model
);
obj2, y = compute_solution_from_selected_columns(instance, paths; bin=true);
obj2

#@info "Col gen" obj

#paths

#dual1
#dual2

#@info "Final" obj2

c_upp, y = compute_solution_from_selected_columns(instance, paths);
paths[[y[p] for p in paths] .== 1.0]

#@info "Final" obj2

#sum(y)

sol = heuristic_solution(instance; nb_it=10000);
heuristic_val = evaluate_solution(sol, instance);
#short(sol)

exact_val, paths = solve_scenarios(instance);

c, _ = compute_solution_from_selected_columns(instance, paths);

if !(c_low ≈ exact_val)
    @warn "🔹" c_low exact_val
end

@info "Gap" c_upp - c_low c_upp - exact_val exact_val c c_upp c_low

#@info "Objectives $seed $nb_tasks $nb_scenarios" c_low c_upp heuristic_val exact_val
