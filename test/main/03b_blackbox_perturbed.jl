using BlackBoxOptim
using Flux
using InferOpt
using JLD2
using Logging
using ProgressMeter
using Random
using StochasticVehicleScheduling
using StochasticVehicleScheduling.Training
using TensorBoardLogger
using UnicodePlots
import Statistics: mean, std

Random.seed!(67);
config = "src/training/config.yaml"
trainer = Trainer(config);

cost(y; instance) = evaluate_solution(y, instance)
loss = PerturbedComposition(
    PerturbedAdditive(trainer.pipeline.maximizer; ε=1000, M=5, seed=0), cost
)

(; X, Y) = trainer.data.train;

mean(X[1].features; dims=2)
std(X[1].features; dims=2)

normalize_data!(X);

mean(X[1].features; dims=2)
std(X[1].features; dims=2)

X_test = trainer.data.test.X;
Y_test = trainer.data.test.Y;

train_cost_opt = [cost(y; instance=x) for (x, y) in zip(X, Y)];
test_cost_opt = [cost(y; instance=x) for (x, y) in zip(X_test, Y_test)];

typeof(X)

#optimal_value = mean(evaluate_solution(y, x) for (x, y) in zip(X, Y))

## Perturbed
nb_features = 20
dim = nb_features + 1
function blackbox_loss(W)
    encoder = Chain(
        Dense(reshape(W[1:nb_features], 1, nb_features), W[(nb_features + 1):end]), vec
    )
    return mean(loss(encoder(x.features); instance=x) for x in X)
end

res1 = bboptimize(
    blackbox_loss; SearchRange=(-1.0, 1.0), NumDimensions=dim, MaxTime=60 * 30
)

best_fitness(res1)
w = best_candidate(res1)
final_encoder = Chain(
    Dense(reshape(w[1:nb_features], 1, nb_features), w[(nb_features + 1):end]), vec
)

train_cost = [
    evaluate_solution(easy_problem(final_encoder(x.features); instance=x), x) for x in X
];
train_cost_gap = mean(
    (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
)

test_cost = [
    evaluate_solution(easy_problem(final_encoder(x.features); instance=x), x) for
    x in X_test
];
test_cost_gap = mean(
    (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(test_cost, test_cost_opt)
)

## Direct
function direct_blackbox_loss(W)
    encoder = Chain(
        Dense(reshape(W[1:nb_features], 1, nb_features), W[(nb_features + 1):end]), vec
    )
    θs = [encoder(x.features) for x in X]
    ys = [easy_problem(θ; instance=x) for (x, θ) in zip(X, θs)]
    return mean(evaluate_solution(y, x) for (x, y) in zip(X, ys))
end

res2 = bboptimize(
    direct_blackbox_loss; SearchRange=(-1.0, 1.0), NumDimensions=dim, MaxTime=60 * 30
)

best_fitness(res2)
w2 = best_candidate(res2)
final_encoder2 = Chain(
    Dense(reshape(w2[1:nb_features], 1, nb_features), w2[(nb_features + 1):end]), vec
)
value2 = mean(
    evaluate_solution(easy_problem(final_encoder2(x.features); instance=x), x) for x in X
)

train_cost = [
    evaluate_solution(easy_problem(final_encoder2(x.features); instance=x), x) for x in X
];
train_cost_gap = mean(
    (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
)

test_cost = [
    evaluate_solution(easy_problem(final_encoder2(x.features); instance=x), x) for
    x in X_test
];
test_cost_gap = mean(
    (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(test_cost, test_cost_opt)
)

###
x = X[1];
regul = PerturbedNormal(easy_problem; ε=1000, M=5)
yr = regul(initial_encoder(x.features); instance=x)
