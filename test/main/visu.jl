using JLD2
using Flux
using StochasticVehicleScheduling


#encoder = load("logs/learn_by_imitation/model_100.jld2")["data"]
encoder = load("logs/test_new_inferopt_normalized_16/model_10000.jld2")["data"]
encoder = Chain(Dense(20 => 1), vec)

data = load("data/data50_normalized/test.jld2");
X = data["X"];
Y = data["Y"];

# pred = zeros(length(X), nb_tasks)
# groundtruth = zeros(length(Y), nb_tasks)

# for (i, (x, y)) in enumerate(zip(X, Y))
#     #x = X[1];
#     θ = encoder(x.features);
#     ypred = easy_problem(θ; instance=x);
#     nb_tasks = 50
#     pred[i, :] = evaluate_solution2(ypred, x)
#     groundtruth[i, :] = evaluate_solution2(y, x)
# end

# size(pred)

# axis = (; title="Groundtruth delays", xlabel="Task index", ylabel="Instance index")
# fig, ax, hm = heatmap(range(1, 50), range(1, 50), groundtruth; axis=axis);
# Colorbar(fig[:, end+1], hm);
# save("fig.png", fig)

# axis2 = (; title = "Predicted delays", xlabel = "Task index", ylabel="Instance index")
# fig2, ax2, hm2 = heatmap(range(1, 50), range(1, 50), pred; axis=axis2);
# Colorbar(fig2[:, end+1], hm2);
# save("fig2.png", fig2)


## ---
using GLMakie

index = 1
x = X[index];
y = Y[index];
ypred = Solution(easy_problem(encoder(x.features); instance=x), x);
_, ypred = solve_deterministic_VSP(x; include_delays=true)

nb_tasks = length(x.city.tasks)-2

start_times = [task.start_time for task in x.city.tasks[2:end-1]]

axis2 = (; title="Groundtruth:", xlabel="Time")
vehiclesgt = [argmax(y.path_value[:, i]) for i in 1:nb_tasks]
groundtruth = evaluate_solution2(y, x)
position = [(i, j) for (i, j) in zip(start_times, vehiclesgt)]
fig2, ax2, hm2 = scatter(position;
    color=groundtruth, markersize=groundtruth, axis=axis2, colormap=:thermal);
#text!(ax2, ["$i" for i in 1:nb_tasks], position=position, font="JuliaMono")
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
Colorbar(fig2[:, end+1], hm2);
save("grountruth.png", fig2)

axis = (; title="Prediction: random", xlabel="Time")
vehiclespred = [argmax(ypred.path_value[:, i]) for i in 1:nb_tasks]
pred = evaluate_solution2(ypred, x)
#position = [(i, j) for (i, j) in zip(start_times, vehiclespred)]
fig, ax, hm = scatter(position;
    color=pred, markersize=pred, axis=axis, colormap=:thermal);
#text!(ax, ["$i" for i in 1:nb_tasks], position=position, font="JuliaMono")
for v in 1:size(y.path_value, 1)
    xs = Float64[]
    ys = Int[]
    for i in 1:50
        if ypred.path_value[v, i] == 1
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
Colorbar(fig[:, end+1], hm);
save("random.png", fig)
