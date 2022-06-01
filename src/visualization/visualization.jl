function plot_instance(instance::Instance)
    (; city) = instance
    fig = Figure()
    ax = Axis(fig[1, 1])
    ax.xlabel = "x"
    ax.ylabel = "y"
    (; tasks, district_width, width) = city
    for (i_task, task) in enumerate(tasks[1:end-1])
        (; start_point, end_point) = task
        points = [(start_point.x, start_point.y), (end_point.x, end_point.y)]
        scatter!(ax, points; markersize=40, marker=:rect)
        lines!(ax, points)
        text!(ax, "$i_task"; position=points[1], align=(:center, :center))
    end
    ticks = 0:district_width:width
    ax.xticks = ticks
    ax.yticks = ticks
    xlims!(ax, [-1, width+1])
    ylims!(ax, [-1, width+1])
    return fig
end

function plot_solution(solution::Solution, instance::Instance)
    (; city) = instance
    (; tasks, district_width, width) = city
    path_list = compute_path_list(solution)
    # fig = plot_instance(instance)
    # ax = contents(fig[1, 1])[1]
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
            # lines!(ax, [x, start_point.x], [y, start_point.y]; color=:black)
            # (; x, y) = end_point
        end
        scatterlines!(ax, X, Y)
    end
    ticks = 0:district_width:width
    ax.xticks = ticks
    ax.yticks = ticks
    xlims!(ax, [-1, width+1])
    ylims!(ax, [-1, width+1])
    return fig
end
