using StochasticVehicleScheduling
using Random

Random.seed!(1)

X_train, Y_train = generate_dataset(1; nb_tasks=200, nb_scenarios=1, nb_it=10_000);
x, y = X_train[1], Y_train[1];

heuristic_value = evaluate_solution(y, x)

#_, deterministic_solution = solve_deterministic_VSP(x)
#deterministic_value = evaluate_solution(deterministic_solution, x)

optimal_value, optimal_solution = solve_scenario(x)
