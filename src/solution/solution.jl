"""
    Solution

Should always be associated with an `Instance`

# Fields
- `value::BitVector`: for each graph edge of instance, 1 if selected, else 0
- `path_value::BitMatrix`: each row represents a vehicle, each column a task.
    1 if task is done by the vehicle, else 0
"""
struct Solution
    value::BitVector       # for every edge of graph 1 if selected, else 0
    path_value::BitMatrix  # list of vehicles paths
end

function get_nb_vehicles(path_value::BitMatrix)
    return sum(any(path_value; dims=2))
end

function get_nb_vehicles(solution::Solution)
    return get_nb_vehicles(solution.path_value)
end

function get_routes(solution::Solution)
    res = Vector{Int}[]
    for vehicle in 1:get_nb_vehicles(solution)
        resv = Int[]
        for (index, value) in enumerate(solution.path_value[vehicle, :])
            if value
                push!(resv, index + 1)
            end
        end
        push!(res, resv)
    end
    return res
end

function Solution(value::BitVector, instance::Instance)
    graph = instance.graph
    nb_tasks = nv(graph)
    is_selected = falses(nb_tasks, nb_tasks)
    for (i, edge) in enumerate(edges(graph))
        if value[i]
            is_selected[edge.src, edge.dst] = true
        end
    end

    path_sol = path_solution_from_JuMP_array(is_selected, graph)
    return Solution(value, path_sol)
end

function Solution(path_value::BitMatrix, instance::Instance)
    graph = instance.graph
    solution = falses(ne(graph))

    mat = to_array(path_value, instance)

    for (edge_index, edge) in enumerate(edges(graph))
        if mat[edge.src, edge.dst]
            solution[edge_index] = true
        end
        # src = edge.src-1
        # dst = edge.dst-1
        # if any(mapslices(x -> src == 0 ? x[dst] : (dst == length(x)+1 ? x[src] : x[src] && x[dst]), path_value, dims=2))
        #     solution[edge_index] = true
        # end
    end

    return Solution(solution, path_value)
end

function solution_from_JuMP_array(x::AbstractArray, graph::AbstractGraph)
    sol = falses(ne(graph)) # init

    for (a, edge) in enumerate(edges(graph))
        if x[edge.src, edge.dst] == 1
            sol[a] = true
        end
    end

    return Solution(sol, path_solution_from_JuMP_array(x, graph))
end

function path_solution_from_JuMP_array(x::AbstractArray, graph::AbstractGraph)
    nb_tasks = nv(graph)
    sol = falses(nb_tasks - 2, nb_tasks - 2) # init
    job_indices = 2:(nb_tasks - 1)

    start = [i for i in job_indices if x[1, i] ≈ 1]
    for (v_index, task) in enumerate(start)
        current_task = task
        while current_task < nb_tasks
            sol[v_index, current_task - 1] = true
            next_tasks = [
                i for i in outneighbors(graph, current_task) if
                isapprox(x[current_task, i], 1; atol=0.1)
            ]
            # TODO : there is a more efficient way to search for next task (but more dangerous)
            if length(next_tasks) == 1
                current_task = next_tasks[1]
            elseif length(next_tasks) == 0
                @warn "No new task :("
                @info current_task
                @info outneighbors(graph, current_task)
                @info [x[current_task, i] for i in outneighbors(graph, current_task)]
                current_task = nb_tasks
            elseif length(next_tasks) > 1
                @warn "Flow constraint is broken..."
                current_task = next_tasks[1]
            end
        end
    end
    return sol
end

function basic_path_solution(graph::AbstractGraph)
    nb_tasks = nv(graph) - 2
    sol = falses(nb_tasks, nb_tasks)
    for i_task in 1:nb_tasks
        sol[i_task, i_task] = true
    end
    return sol
end

function basic_solution(instance::Instance)
    graph = instance.graph
    value = falses(ne(graph))

    for (a, edge) in enumerate(edges(graph))
        if edge.src == 1 || edge.dst == nv(graph)
            value[a] = true
        end
    end

    return Solution(value, basic_path_solution(graph))
end

function evaluate_task(
    i_task::Integer,
    instance::Instance,
    old_task_index::Integer,
    old_delay::Real,
    scenario::Int,
)
    (; slacks, delays) = instance

    delay = delays[i_task, scenario]
    slack = slacks[old_task_index, i_task][scenario]

    return delay + max(old_delay - slack, 0)
end

function evaluate_scenario(path_value::BitMatrix, instance::Instance, scenario_index::Int)
    total_delay = 0.0
    nb_tasks = instance.city.nb_tasks

    for i_vehicle in 1:nb_tasks
        # no delay if no tasks
        if !any(@view path_value[i_vehicle, :])
            continue
        end

        task_delay = 0.0
        old_task_index = 1 # always start at depot

        for i_task in 1:nb_tasks
            # check if task is done by this vehicle
            if !path_value[i_vehicle, i_task]
                continue
            end
            task_delay = evaluate_task(
                i_task + 1, instance, old_task_index, task_delay, scenario_index
            )
            old_task_index = i_task + 1

            total_delay += task_delay
        end
    end
    return total_delay
end

function evaluate_scenario2(path_value::BitMatrix, instance::Instance, scenario_index::Int)
    total_delay = 0.0
    nb_tasks = instance.city.nb_tasks

    task_delays = zeros(nb_tasks)

    for i_vehicle in 1:nb_tasks
        # no delay if no tasks
        if !any(@view path_value[i_vehicle, :])
            continue
        end

        task_delay = 0.0
        old_task_index = 1 # always start at depot

        for i_task in 1:nb_tasks
            # check if task is done by this vehicle
            if !path_value[i_vehicle, i_task]
                continue
            end
            task_delay = evaluate_task(
                i_task + 1, instance, old_task_index, task_delay, scenario_index
            )
            old_task_index = i_task + 1

            total_delay += task_delay
            task_delays[i_task] = task_delay
        end
    end
    return task_delays
end

function evaluate_scenario(solution::Solution, instance::Instance, scenario_index::Int)
    return evaluate_scenario(solution.path_value, instance, scenario_index)
end

function evaluate_solution2(path_value::BitMatrix, instance::Instance)
    nb_scenarios = get_nb_scenarios(instance)

    nb_tasks = instance.city.nb_tasks
    average_delay = zeros(nb_tasks)
    for s in 1:nb_scenarios
        average_delay .+= evaluate_scenario2(path_value, instance, s)
    end
    average_delay ./= nb_scenarios

    return average_delay
end

function evaluate_solution(path_value::BitMatrix, instance::Instance)
    nb_scenarios = get_nb_scenarios(instance)

    average_delay = 0.0
    for s in 1:nb_scenarios
        average_delay += evaluate_scenario(path_value, instance, s)
    end
    average_delay /= nb_scenarios

    nb_vehicles = sum(any(path_value; dims=2))
    return instance.city.vehicle_cost * nb_vehicles +
           instance.city.delay_cost * average_delay
end

function evaluate_solution(solution::Solution, instance::Instance)
    return evaluate_solution(solution.path_value, instance)
end

function evaluate_solution2(solution::Solution, instance::Instance)
    return evaluate_solution2(solution.path_value, instance)
end

function evaluate_solution(value::BitVector, instance::Instance)
    return evaluate_solution(Solution(value, instance), instance)
end

function evaluate_solution2(value::BitVector, instance::Instance)
    return evaluate_solution2(Solution(value, instance), instance)
end

# function evaluate_solution(value::AbstractVector, instance::Instance)
#     return evaluate_solution(Solution(convert(BitVector, value), instance), instance)
# end

function to_array(path_value::BitMatrix, instance::Instance)
    graph = instance.graph
    nb_nodes = nv(graph)
    nb_tasks = nb_nodes - 2
    job_indices = 2:(nb_nodes - 1)
    mat = falses(nb_nodes, nb_nodes)

    # check each task used once and only once
    for i in 1:nb_tasks
        if !any(@view path_value[i, :])
            continue
        end
        # else
        current_task = 1
        while true
            index_shift = find_first_one(@view path_value[i, current_task:end])
            if isnothing(index_shift)
                mat[current_task, nb_nodes] = true
                break
            end
            next_task = current_task + index_shift
            if !has_edge(graph, current_task, next_task)
                @warn "Flow not respected" current_task next_task
                @warn "" outneighbors(graph, current_task)
                return false
            end
            mat[current_task, next_task] = true
            current_task = next_task
        end
    end

    return mat
end

"""
    to_array(solution::Solution, instance::Instance)

Returns a BitMatrix, with value true at each index (i, j) if corresponding edge of graph
is selected in the solution
"""
function to_array(solution::Solution, instance::Instance)
    return to_array(solution.path_value, instance)
end

function is_admissible(solution::Solution, instance::Instance)
    graph = instance.graph
    nb_nodes = nv(graph)
    nb_tasks = nb_nodes - 2
    job_indices = 2:(nb_nodes - 1)
    mat = falses(nb_nodes, nb_nodes)

    # check each task used once and only once
    for i in 1:nb_tasks
        if !any(@view solution.path_value[i, :])
            continue
        end
        # else
        current_task = 1
        while true
            index_shift = find_first_one(@view solution.path_value[i, current_task:end])
            if isnothing(index_shift)
                mat[current_task, nb_nodes] = true
                break
            end
            next_task = current_task + index_shift
            if !has_edge(graph, current_task, next_task)
                @warn "Flow not respected" current_task next_task
                @warn "" outneighbors(graph, current_task)
                return false
            end
            mat[current_task, next_task] = true
            current_task = next_task
        end
    end

    if !all(sum(solution.path_value; dims=1) .== 1)
        @warn "One task done by more than one vehicle"
        return false
    end

    for i in job_indices
        s1 = sum(mat[j, i] for j in inneighbors(graph, i))
        s2 = sum(mat[i, j] for j in outneighbors(graph, i))
        if s1 != s2 || s1 != 1
            @warn "Flow is broken" i s1 s2
            @warn "" inneighbors(graph, i)
            @warn "" [mat[j, i] for j in inneighbors(graph, i)]
            @warn "" outneighbors(graph, i)
            @warn "" [mat[i, j] for j in outneighbors(graph, i)]
            @warn "" mat
            @warn "" solution.path_value
            return false
        end
    end

    return true
end

function compute_path_list(solution::Solution)
    (; path_value) = solution
    paths = Vector{Int64}[]
    for v in 1:size(path_value, 1)
        path = [1]
        for (i, elem) in enumerate(path_value[v, :])
            if elem == 1
                push!(path, i + 1)
            end
        end
        push!(path, size(path_value, 2) + 2)
        push!(paths, path)
    end
    return paths
end
