using Flux
using GLMakie
using JLD2
using StochasticVehicleScheduling

function visualize_solution(x, y, start_times, vehiclesgt; output_file="groundtruth.png")
    obj = evaluate_solution(y, x)
    axis2 = (;
        title="Total objective value: $(round(obj, digits=2))",
        xlabel="Time",
        yticklabelsvisible=false,
        xgridvisible=false,
        ygridvisible=false,
        yticksvisible=false,
    )
    groundtruth = evaluate_solution2(y, x)
    position = [(i, j) for (i, j) in zip(start_times, vehiclesgt)]
    fig2, axis, hm2 = scatter(position;
        color=groundtruth, markersize=groundtruth, axis=axis2, colormap=:thermal);
    for v in 1:size(y.path_value, 1)
        xs = Float64[]
        ys = Int[]
        for i in 1:50
            if y.path_value[v, i] == 1
                push!(xs, start_times[i])
                push!(ys, vehiclesgt[i])
            end
        end
        if length(xs) == 0
            break
        end
        for i in 1:length(xs)-1
            arrows!([xs[i]], [ys[i]], [xs[i+1] - xs[i]], [ys[i+1] - ys[i]];
                arrowsize=15, lengthscale=1.0)
        end
        #lines!(xs, ys, color=:black)
    end
    Colorbar(fig2[:, end+1], hm2, ticks = 0:5:60);
    save(output_file, fig2)
    return
end

# encoder = load("logs/test_new_inferopt_normalized_16/model_10000.jld2")["data"]
function main(index)
    encoder = Chain(Dense(20 => 1, bias=false), vec)

    best = load("final_experiments/imitation_50tasks50scenarios/model_50.jld2")
    encoder = best["data"]
    σ = best["σ"]
    encoder[1].weight' ./= σ

    best2 = load("final_experiments/experience_50tasks50scenarios/best.jld2")
    encoder2 = best2["data"]
    σ2 = best2["σ"]
    encoder2[1].weight' ./= σ2

    data = load("data/50tasks50scenarios/train.jld2");
    X = data["X"];
    Y = data["Y"];

    x = X[index];
    y = Y[index];
    _, ypred = solve_deterministic_VSP(x; include_delays=true)
    ypred1 = Solution(easy_problem(encoder(x.features); instance=x), x);
    ypred2 = Solution(easy_problem(encoder2(x.features); instance=x), x);

    nb_tasks = length(x.city.tasks)-2
    start_times = [task.start_time for task in x.city.tasks[2:end-1]]
    vehiclesgt = [argmax(y.path_value[:, i]) for i in 1:nb_tasks]
    visualize_solution(x, y, start_times, vehiclesgt; output_file="figures/local_search.png")
    #visualize_solution(x, ypred, start_times, vehiclesgt; output_file="figures/local_search.png")
    visualize_solution(x, ypred1, start_times, vehiclesgt; output_file="figures/imitation.png")
    visualize_solution(x, ypred2, start_times, vehiclesgt; output_file="figures/experience.png")
end

main(42)

evaluate_solution(ypred2, x)
evaluate_solution(y, x)

9000 + 2*sum(evaluate_solution2(ypred2, x))
9000 + 2*sum(evaluate_solution2(y, x))

StochasticVehicleScheduling.get_nb_vehicles(y)
StochasticVehicleScheduling.get_nb_vehicles(ypred2)

is_admissible(ypred, x)

ypred2.path_value
