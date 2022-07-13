using JLD2
using Flux
using StochasticVehicleScheduling
using StochasticVehicleScheduling.Training
using Statistics: mean
using ProgressMeter

log_dirs = [
    #"imitation_25tasks10scenarios_exact_1",
    # "imitation_25tasks10scenarios",
    # "imitation_25tasks10scenarios_garbage",
    "imitation_25tasks10scenarios_exact",
    "imitation_50tasks50scenarios",
    "imitation_100tasks50scenarios",
    # "experience_25tasks10scenarios_exact",
    "experience_25tasks10scenarios_exact_2",
    "experience_50tasks50scenarios_2",
    "experience_100tasks50scenarios_2",
    # "imitation_mixed",
]

# log_dirs = [
#     "experience_25tasks10scenarios_exact",
#     "experience_50tasks50scenarios",
#     "experience_100tasks50scenarios",
# ]

test_datasets = [
    "25tasks10scenarios_exact_uncentered",
    "50tasks50scenarios_uncentered",
    "100tasks50scenarios_uncentered",
    "200tasks50scenarios_uncentered",
    #"mixed",
]

function evaluate_model(model_dir, test_data)
    log_dir = joinpath("final_experiments", model_dir)
    model_file = joinpath(log_dir, "best.jld2")
    config_file = joinpath(log_dir, "config.yaml")
    trainer = Trainer(config_file; create_logger=false);

    best_model = load(model_file)
    encoder = best_model["data"]
    encoder[1].weight
    step = best_model["epoch"]

    data_dir = joinpath("data", test_data)
    data_test = load(joinpath(data_dir, "test.jld2"));
    X_test = data_test["X"];
    Y_test = data_test["Y"];

    Y_pred = [easy_problem(encoder(x.features); instance=x, model_builder=grb_model) for x in X_test];

    (; cost) = trainer;
    train_cost = [cost(y; instance=x) for (x, y) in zip(X_test, Y_pred)];
    train_cost_opt = [cost(y; instance=x) for (x, y) in zip(X_test, Y_test)];

    average_cost_gap = mean(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
    ) * 100
    max_cost_gap = maximum(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
    ) * 100
    average_cost_per_task = mean(
        c / x.city.nb_tasks for (c, x) in zip(train_cost, X_test)
    )
    @info "$log_dir -> $test_data" step average_cost_gap max_cost_gap average_cost_per_task
end

function evaluate_cost(model_dir, test_data)
    log_dir = joinpath("final_experiments", model_dir)
    model_file = joinpath(log_dir, "best.jld2")
    config_file = joinpath(log_dir, "config.yaml")
    trainer = Trainer(config_file; create_logger=false);

    best_model = load(model_file)
    encoder = best_model["data"]
    encoder[1].weight
    step = best_model["epoch"]

    data_dir = joinpath("data", test_data)
    data_test = load(joinpath(data_dir, "test.jld2"));

    (; cost) = trainer;
    c_sum = 0
    c = 0
    @showprogress for x in data_test["X"]
        c += 1
        c_sum += cost(easy_problem(encoder(x.features); instance=x, model_builder=grb_model); instance=x) / x.city.nb_tasks
    end

    average_cost_per_task = c_sum / c
    @info "$log_dir -> $test_data" step average_cost_per_task
end

for log_dir in log_dirs
    for test_data in test_datasets
        evaluate_model(log_dir, test_data)
    end
    #evaluate_cost(log_dir, "1000tasks10scenarios")
    println("---")
end
