function experiment(seed::Int)
    Random.seed!(seed)

    X_train, Y_train = generate_dataset(1; nb_tasks=20, nb_scenarios=1, nb_it=10_000);
    x, y = X_train[1], Y_train[1]

    heuristic_value = evaluate_solution(y, x)

    _, deterministic_solution = solve_deterministic_VSP(x)
    deterministic_value = evaluate_solution(deterministic_solution, x)

    optimal_value, optimal_solution = solve_scenario(x)

    @test heuristic_value <= deterministic_value
    @test optimal_value <= heuristic_value
    @test optimal_value <= deterministic_value
    @test optimal_value ≈ evaluate_solution(optimal_solution, x)
end

## Experiment

for i in 1:50
    experiment(i)
end
