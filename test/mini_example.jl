using StochasticVehicleScheduling
const VSP = StochasticVehicleScheduling

function create_mini_instance(delay::Float64)
    width = 50
    nb_scenarios = 1

    city_center = VSP.Point(25., 25.)
    tasks = VSP.Task[]

    push!(tasks, VSP.Task(VSP.depot_start, city_center, city_center, 0.0, 0.0, VSP.ZERO_UNIFORM, [0.0], [0.0]))
    final_task_time = 24 * 60.0
    push!(tasks, VSP.Task(VSP.depot_start, city_center, city_center, final_task_time, final_task_time, VSP.ZERO_UNIFORM, [final_task_time], [final_task_time]))

    p1 = VSP.Point(25., 30.)
    push!(tasks, VSP.Task(VSP.depot_start, p1, p1, 60, 120, VSP.ZERO_UNIFORM, [60.0], [120.0+delay]))

    p2 = VSP.Point(25., 35.)
    push!(tasks, VSP.Task(VSP.depot_start, p2, p2, 130, 140, VSP.ZERO_UNIFORM, [125.0], [140.0]))

    VSP.distance(p1, p2)

    districts = Matrix{VSP.District}(undef, 1, 1)
    districts[1, 1] = VSP.District(VSP.ZERO_UNIFORM, zeros(nb_scenarios, 24))
    district_width = width

    city = VSP.City(;
        nb_scenarios=nb_scenarios,
        width=width,
        vehicle_cost=1000,
        nb_tasks=length(tasks)-2,
        tasks=tasks,
        district_width=district_width,
        districts=districts,
        delay_cost=2,
        random_inter_area_factor=VSP.ZERO_UNIFORM,
        scenario_inter_area_factor=zeros(nb_scenarios, 24)
    )
    sort!(city.tasks, by=task -> task.start_time, rev=false)

    instance = Instance(city)
    return instance
end

instance = create_mini_instance(0.0)
v, sol = solve_deterministic_VSP(instance)
delay = VSP.evaluate_scenario(sol, instance, 1)
@test delay == 0.0

instance = create_mini_instance(10.0)
v, sol = solve_deterministic_VSP(instance)
delay = VSP.evaluate_scenario(sol, instance, 1)
@test delay == 10.0
