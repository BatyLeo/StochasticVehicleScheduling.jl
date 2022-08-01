using Flux
using InferOpt
using Optim
using Random
using Statistics: mean
using StochasticVehicleScheduling
using StochasticVehicleScheduling.Training
using JLD2

Random.seed!(67);

data = load("data/25tasks10scenarios/train.jld2");
X, Y = data["X"], data["Y"];

data_val = load("data/25tasks10scenarios/validation.jld2");
X_val, Y_val = data_val["X"], data_val["Y"];

cost(y; instance) = evaluate_solution(y, instance)
train_cost_opt = [cost(y; instance=x) for (x, y) in zip(X, Y)];
val_cost_opt = [cost(y; instance=x) for (x, y) in zip(X_val, Y_val)];

function evaluate(w)
    final_encoder = Chain(Dense(reshape(w, 1, nb_features)), vec)

    train_cost = [
        evaluate_solution(easy_problem(final_encoder(x.features); instance=x), x) for x in X
    ]
    train_cost_gap = mean(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
    )

    val_cost = [
        evaluate_solution(easy_problem(final_encoder(x.features); instance=x), x) for
        x in X_val
    ]
    val_cost_gap = mean(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(val_cost, val_cost_opt)
    )

    return train_cost_gap, val_cost_gap
end

#optimal_value = mean(evaluate_solution(y, x) for (x, y) in zip(X, Y))

maximizer(θ::AbstractVector; instance) = easy_problem(θ; instance, model_builder=grb_model)
loss = ProbabilisticComposition(
    PerturbedAdditive(maximizer; ε=100, nb_samples=5, seed=0), cost
)

nb_features = 20
function bbloss(W::Vector)
    encoder = Chain(Dense(reshape(W, 1, nb_features)), vec)
    return mean(loss(encoder(x.features); instance=x) for x in X)
end

w_opt = Optim.minimizer(optimize(bbloss, randn(nb_features), BFGS()))
@info bbloss(w_opt)
@info evaluate(w_opt)

# ----

## Direct
function direct_blackbox_loss(W::Vector)
    encoder = Chain(Dense(reshape(W, 1, nb_features)), vec)
    θs = [encoder(x.features) for x in X]
    ys = [easy_problem(θ; instance=x) for (x, θ) in zip(X, θs)]
    return mean(evaluate_solution(y, x) for (x, y) in zip(X, ys))
end

w_opt2 = Optim.minimizer(optimize(direct_blackbox_loss, randn(nb_features), BFGS()))
@info bbloss(w_opt2)
@info evaluate(w_opt2)
