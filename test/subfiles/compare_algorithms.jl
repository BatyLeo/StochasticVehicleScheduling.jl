function experiment(seed::Int)
    Random.seed!(seed)
    # Create a random instance and compute its heuristic solution
    x = Instance(create_random_city(; nb_tasks=20, nb_scenarios=10))
    y = heuristic_solution(x; nb_it=10_000)
    local_search_value = evaluate_solution(y, x)

    # Heuristic value (should be worse than local search, since local search is initialized with it)
    _, deterministic_solution = solve_deterministic_VSP(x)
    deterministic_value = evaluate_solution(deterministic_solution, x)

    # Optimal value
    optimal_value, optimal_solution = solve_scenarios(x)

    # Column generation approach
    _, _, relaxation_paths, _, _ = column_generation(x; only_relaxation=true)
    _, _, full_paths, _, _ = column_generation(x; only_relaxation=false)

    _, _, sol1 = compute_solution_from_selected_columns(x, relaxation_paths; bin=true)
    relaxation_bin_value = evaluate_solution(solution_from_paths(sol1, x), x)

    _, _, sol2 = compute_solution_from_selected_columns(x, relaxation_paths; bin=false)
    relaxation_value = evaluate_solution(solution_from_paths(sol2, x), x)

    _, _, sol3 = compute_solution_from_selected_columns(x, full_paths; bin=true)
    full_value = evaluate_solution(solution_from_paths(sol3, x), x)

    # Check admissibility of solutions
    @test is_admissible(y, x)
    @test is_admissible(deterministic_solution, x)
    @test is_admissible(optimal_solution, x)
    # Check that local serach has not deteriorated the initial solution
    @test local_search_value <= deterministic_value
    # Check that local serach is not better than optimal solution
    @test optimal_value ≈ local_search_value || optimal_value <= local_search_value
    @test optimal_value ≈ evaluate_solution(optimal_solution, x)
    # Column generation checks
    @test relaxation_value <= relaxation_bin_value
    @test full_value <= relaxation_bin_value
    @test relaxation_value <= full_value
    @test full_value ≈ optimal_value
end

for i in 1:20
    experiment(i)
end
