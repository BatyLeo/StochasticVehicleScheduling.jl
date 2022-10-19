"""
    plot_instance_on_map(instance::Instance)

Draw representation of `instance` on its map.
"""
function plot_instance_on_map(instance::Instance)
    (; city) = instance
    fig = Figure()
    ax = Axis(fig[1, 1])
    ax.xlabel = "x"
    ax.ylabel = "y"
    (; tasks, district_width, width) = city
    for (i_task, task) in enumerate(tasks[1:(end - 1)])
        (; start_point, end_point) = task
        points = [(start_point.x, start_point.y), (end_point.x, end_point.y)]
        scatter!(ax, points; markersize=40, marker=:rect)
        lines!(ax, points)
        text!(ax, "$(i_task-1)"; position=points[1], align=(:center, :center))
    end
    ticks = 0:district_width:width
    ax.xticks = ticks
    ax.yticks = ticks
    xlims!(ax, [-1, width + 1])
    ylims!(ax, [-1, width + 1])
    return fig
end

"""
    plot_instance_on_map(instance::Instance, savepath::String)

Draw representation of `instance` on its map, and save it as a png file savepath.
"""
function plot_instance_on_map(instance::Instance, savepath::String)
    fig = plot_instance_on_map(instance)
    save(savepath, fig)
    return nothing
end

"""
    plot_solution_on_map(solution::Solution, instance::Instance)

Draw representation of `solution` of `instance` on its map.
"""
function plot_solution_on_map(solution::Solution, instance::Instance)
    (; city) = instance
    (; tasks, district_width, width) = city
    path_list = compute_path_list(solution)
    fig = Figure()
    ax = Axis(fig[1, 1])
    for path in path_list
        X = Float64[]
        Y = Float64[]
        (; start_point, end_point) = tasks[path[1]]
        (; x, y) = end_point
        push!(X, x)
        push!(Y, y)
        for task in path[2:end]
            (; start_point, end_point) = tasks[task]
            push!(X, start_point.x)
            push!(Y, start_point.y)
            push!(X, end_point.x)
            push!(Y, end_point.y)
        end
        scatterlines!(ax, X, Y)
    end
    ticks = 0:district_width:width
    ax.xticks = ticks
    ax.yticks = ticks
    xlims!(ax, [-1, width + 1])
    ylims!(ax, [-1, width + 1])
    return fig
end

"""
    plot_solution_on_map(solution:Solution, instance::Instance, savepath)

Draw representation of `solution` of `instance` on its map, and save it as png file savepath.
"""
function plot_solution_on_map(solution::Solution, instance::Instance, savepath::String)
    fig = plot_solution_on_map(solution, instance)
    save(savepath, fig)
    return nothing
end
