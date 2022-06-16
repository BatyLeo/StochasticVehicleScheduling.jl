using Flux
using InferOpt
using NLopt
using Random
using Statistics: mean
using StochasticVehicleScheduling
using StochasticVehicleScheduling.Training

Random.seed!(67);
config = "src/training/config.yaml"
trainer = Trainer(config);

cost(y; instance) = evaluate_solution(y, instance)
loss = PerturbedCost(PerturbedNormal(trainer.pipeline.maximizer; ε=1000, M=5), cost)

(; X, Y) = trainer.data.train;

X_test = trainer.data.test.X;
Y_test = trainer.data.test.Y;

train_cost_opt = [cost(y; instance=x) for (x, y) in zip(X, Y)];
test_cost_opt = [cost(y; instance=x) for (x, y) in zip(X_test, Y_test)];

function evaluate(w)
    final_encoder = Chain(Dense(reshape(w[1:nb_features], 1, nb_features), w[nb_features+1:end]), vec)

    train_cost = [evaluate_solution(easy_problem(final_encoder(x.features); instance=x), x) for x in X];
    train_cost_gap = mean(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
    )

    test_cost = [evaluate_solution(easy_problem(final_encoder(x.features); instance=x), x) for x in X_test];
    test_cost_gap = mean(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(test_cost, test_cost_opt)
    )

    return train_cost_gap, test_cost_gap
end

#optimal_value = mean(evaluate_solution(y, x) for (x, y) in zip(X, Y))

## Perturbed
nb_features = 20
dim = nb_features + 1
function blackbox_loss(W::Vector, g)
    encoder = Chain(Dense(reshape(W[1:nb_features], 1, nb_features), W[nb_features+1:end]), vec)
    return mean(loss(encoder(x.features); instance=x) for x in X)
end

opt = Opt(:GN_DIRECT_L, dim)
maxtime!(opt, 60 * 30)
lower_bounds!(opt, ones(dim) * -1)
upper_bounds!(opt, ones(dim))
min_objective!(opt, blackbox_loss)
value, w_opt, status = optimize(opt, zeros(dim))

@info "perturbed" w_opt
@info evaluate(w_opt)

## Direct
function direct_blackbox_loss(W::Vector, g)
    encoder = Chain(Dense(reshape(W[1:nb_features], 1, nb_features), W[nb_features+1:end]), vec)
    θs = [encoder(x.features) for x  in X]
    ys = [easy_problem(θ; instance=x) for (x, θ) in zip(X, θs)]
    return mean(evaluate_solution(y, x) for (x, y) in zip(X, ys))
end

opt = Opt(:GN_DIRECT_L, dim)
maxtime!(opt, 60 * 10)
lower_bounds!(opt, ones(dim) * -1)
upper_bounds!(opt, ones(dim))
min_objective!(opt, direct_blackbox_loss)
value, w_opt2, status = optimize(opt, zeros(dim))

@info "unperturbed" w_opt2
@info evaluate(w_opt2)
