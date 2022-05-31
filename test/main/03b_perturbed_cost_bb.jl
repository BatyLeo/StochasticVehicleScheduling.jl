using BlackBoxOptim
using InferOpt
using InferOpt.Testing
using JLD2
using Logging
using ProgressMeter
using Random
using StochasticVehicleScheduling
using TensorBoardLogger
using UnicodePlots
import Statistics: mean

Random.seed!(67);

## Dataset
dataset_path = "data/data50.jld2"
data = load(dataset_path)["data"];

## GLM model and loss
nb_features = 20
cost(y; instance) = evaluate_solution(y, instance)
loss = PerturbedCost(PerturbedNormal(easy_problem; ε=100, M=5), cost)


(; X, Y) = data.train;
#solve_scenarios(X[1])

optimal_value = mean(evaluate_solution(y, x) for (x, y) in zip(X, Y))

initial_encoder = Chain(Dense(nb_features => 1), vec)
θ = initial_encoder(x.features)
y_pred = easy_problem(θ; instance=x)
evaluate_solution(y_pred, x)

function blackbox_loss(W)
    encoder = Chain(Dense(reshape(W, 1, nb_features)), vec)
    return mean(loss(encoder(x.features); instance=x) for x in X)
end

res1 = bboptimize(blackbox_loss; SearchRange=(-1.0, 1.0), NumDimensions=nb_features)

best_fitness(res1)
w = best_candidate(res1)
final_encoder = Chain(Dense(reshape(w, 1, nb_features)), vec)
value = mean(evaluate_solution(easy_problem(final_encoder(x.features); instance=x), x) for x in X)

function direct_blackbox_loss(W)
    encoder = Chain(Dense(reshape(W, 1, nb_features)), vec)
    θs = [encoder(x.features) for x  in X]
    ys = [easy_problem(θ; instance=x) for (x, θ) in zip(X, θs)]
    return mean(evaluate_solution(y, x) for (x, y) in zip(X, ys))
end

res2 = bboptimize(direct_blackbox_loss; SearchRange=(-1.0, 1.0), NumDimensions=nb_features)

best_fitness(res2)
w2 = best_candidate(res2)
final_encoder2 = Chain(Dense(reshape(w2, 1, nb_features)), vec)
value2 = mean(evaluate_solution(easy_problem(final_encoder2(x.features); instance=x), x) for x in X)

x = X[1];
regul = PerturbedNormal(easy_problem; ε=1000, M=5)
yr = regul(initial_encoder(x.features); instance=x)
