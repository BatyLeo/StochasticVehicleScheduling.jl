Random.seed!(67)
instance = Instance(
    create_random_city(; nb_tasks=50, nb_scenarios=500, vehicle_cost=1000, delay_cost=2)
)

nb_it = 1_000

@testset "Local search from scratch" begin
    sol = basic_solution(instance)

    sol_value = evaluate_solution(sol, instance)
    @info "Initial solution" short(sol) sol_value

    best_sol, best_sol_value, history_x, history_y = local_search(
        sol, instance; nb_it=nb_it
    )
    @info "Final solution" short(best_sol) best_sol_value

    println(
        scatterplot(history_x, history_y; title="Best objective found", xlabel="Iteration")
    )

    @test sol_value >= best_sol_value
end

@testset "Local search initialized with PLNE" begin
    value, sol = solve_deterministic_VSP(instance; include_delays=false)
    value = evaluate_solution(sol, instance)
    @info "PLNE vehicles" short(sol) value

    value, sol = solve_deterministic_VSP(instance)
    value = evaluate_solution(sol, instance)
    @info "PLNE vehicles + delays" short(sol) value

    best_sol, best_sol_value, history_x, history_y = local_search(
        sol, instance; nb_it=nb_it
    )
    @info "Final solution" short(best_sol) best_sol_value

    println(
        scatterplot(history_x, history_y; title="Best objective found", xlabel="Iteration")
    )

    @test value >= best_sol_value
end
