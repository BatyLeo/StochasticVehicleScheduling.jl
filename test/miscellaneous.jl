function experiment(seed::Int)
    Random.seed!(seed)
    # Create a random instance and compute its heuristic solution
    x = Instance(create_random_city(; nb_tasks=20, nb_scenarios=10))
    y = heuristic_solution(x; nb_it=10_000)

    # Value of y
    local_search_value = evaluate_solution(y, x)

    # Heuristic value (should be worse than local search, since local search is initialized with it)
    _, deterministic_solution = solve_deterministic_VSP(x)
    deterministic_value = evaluate_solution(deterministic_solution, x)

    optimal_value, optimal_solution = solve_scenarios(x)

    @test local_search_value <= deterministic_value
    @test optimal_value ≈ local_search_value || optimal_value <= local_search_value
    @test optimal_value <= deterministic_value
    @test optimal_value ≈ evaluate_solution(optimal_solution, x)
end

## Experiment

for i in 1:50
    experiment(i)
end
