## Imports useful packages
using InferOptModelZoo.VSP, InferOpt
using Random, Test
using Flux
using UnicodePlots
using ProgressMeter
using Statistics: mean
using Logging, TensorBoardLogger


function train_test_split(X::AbstractVector, Y::AbstractVector, split_ratio::Real)
    @assert length(X) == length(Y) "X and Y have different lengths"
    nb_training_samples = length(X) - trunc(Int, length(X) * split_ratio)
    X_train, Y_train = X[1:nb_training_samples], Y[1:nb_training_samples]
    X_test, Y_test = X[nb_training_samples+1:end], Y[nb_training_samples+1:end]
    return (X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
end

function main(run_name::String="run")
    Random.seed!(1);
    lg = TBLogger("tensorboard_logs/$run_name", min_level=Logging.Info)
    ## Dataset creation

    nb_samples = 100
    split_ratio = 0.5
    nb_tasks = 50
    nb_scenarios = 1

    X, Y = generate_dataset(nb_samples; nb_tasks=nb_tasks, nb_scenarios=nb_scenarios, nb_it=10_000);
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, split_ratio)
    #X_test, Y_test = X_train, Y_train  #
    nb_features = size(X_train[1].features, 1)
    data_train, data_test = X_train, X_test;

    ## Initialize the GLM predictor:
    model = Chain(Dense(nb_features => 1), vec)
    pipeline(x) = easy_problem(model(x.features), instance=x);

    initial_pred = [pipeline(x) for x in X_test]
    initial_obj = [evaluate_solution(y, x) for (x, y) in zip(X_test, initial_pred)]
    initial_mean_obj = mean(initial_obj)
    ground_truth_obj = [evaluate_solution(y, x) for (x, y) in zip(X_test, Y_test)]
    ground_truth_mean_obj = mean(ground_truth_obj)

    initial_objective_gap = mean(v_pred - v for (v, v_pred) in zip(ground_truth_obj, initial_obj))

    @info "Ground truth" ground_truth_mean_obj
    @info "Initial" initial_mean_obj
    @info "Difference" initial_objective_gap

    ε = 1.
    M = 5
    cost(y; instance) = evaluate_solution(y, instance)
    loss = PerturbedCost(easy_problem, cost; ε=ε, M=M)
    flux_loss(x) = loss(model(x.features); instance=x)
    opt = ADAM(1e-3);

    nb_epochs = 1000
    hamming_distance(x::AbstractVector, y::AbstractVector) = sum(x[i] != y[i] for i in eachindex(x))
    log_every = 10

    ## Train loop
    @showprogress for epoch in 0:nb_epochs
        "Epoch $epoch"
        if epoch % log_every == 0
            l = mean(flux_loss(x) for x in data_train)
            l_test = mean(flux_loss(x) for x in data_test)
            Y_train_pred = [easy_problem(model(x.features); instance=x) for x in  X_train]
            train_values = [evaluate_solution(y, x) for (x, y) in zip(X_train, Y_train_pred)]
            V_train = mean(v_pred - v for (v_pred, v) in zip(train_values, ground_truth_obj))

            Y_pred = [easy_problem(model(x.features); instance=x) for x in  X_test]
            values = [evaluate_solution(y, x) for (x, y) in zip(X_test, Y_pred)]
            V = mean(v_pred - v for (v_pred, v) in zip(values, ground_truth_obj))
            with_logger(lg) do
                @info "train" loss=l objective_gap=V_train log_step_increment=(epoch== 0 ? 0 : log_every)
                @info "test" loss=l_test objective_gap=V log_step_increment=0
            end
        end

        Flux.train!(flux_loss, Flux.params(model), data_train, opt)
    end
    return nothing
end

main("eps1lr1e-3nonormalise")
