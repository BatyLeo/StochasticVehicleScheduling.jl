using JLD2
using Flux
using StochasticVehicleScheduling
using Statistics: mean
using ProgressMeter
try
    using Gurobi
    @info "Gurobi loaded with success !"
catch
    @info "Gurobi not found, will use Cbc instead."
end

const logdir = "logs"
const data_dir = "data"

log_dirs = [
    "imitation_25tasks10scenarios",
    "imitation_50tasks50scenarios",
    "imitation_100tasks50scenarios",
    "experience_25tasks10scenarios",
    "experience_50tasks50scenarios",
    "experience_100tasks50scenarios",
]

test_datasets = [
    "25tasks10scenarios",
    "50tasks50scenarios",
    "100tasks50scenarios",
]

cost_only = [
    "200tasks50scenarios",
    "300tasks10scenarios",
    "500tasks10scenarios",
    "750tasks10scenarios",
    "1000tasks10scenarios",
]

function evaluate_model(model_dir, test_data)
    log_dir = joinpath(logdir, model_dir)
    model_file = joinpath(log_dir, "best.jld2")
    config_file = joinpath(log_dir, "config.yaml")
    trainer = Trainer(config_file; create_logger=false)

    best_model = load(model_file)
    encoder = best_model["data"]
    σ = best_model["σ"]
    encoder[1].weight ./= reshape(σ, 1, 20)  # rescale

    step = best_model["epoch"]

    data = joinpath(data_dir, test_data)
    data_test = load(joinpath(data, "test.jld2"))
    X_test = data_test["X"]
    Y_test = data_test["Y"]

    model_builder = cbc_model
    try
        model_builder = grb_model
    catch
    end
    Y_pred = [
        easy_problem(encoder(x.features); instance=x, model_builder=model_builder) for
        x in X_test
    ]

    (; cost) = trainer
    train_cost = [cost(y; instance=x) for (x, y) in zip(X_test, Y_pred)]
    train_cost_opt = [cost(y; instance=x) for (x, y) in zip(X_test, Y_test)]

    average_cost_gap =
        mean((c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)) *
        100
    max_cost_gap =
        maximum(
            (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
        ) * 100
    average_cost_per_task = mean(c / x.city.nb_tasks for (c, x) in zip(train_cost, X_test))
    @info "$log_dir -> $test_data" step average_cost_gap max_cost_gap average_cost_per_task
    return Dict(
        "step" => step,
        "average_cost_gap" => average_cost_gap,
        "max_cost_gap" => max_cost_gap,
        "average_cost_per_task" => average_cost_per_task,
    )
end

function evaluate_cost(model_dir, test_data)
    log_dir = joinpath(logdir, model_dir)
    model_file = joinpath(log_dir, "best.jld2")
    config_file = joinpath(log_dir, "config.yaml")
    trainer = Trainer(config_file; create_logger=false)

    best_model = load(model_file)
    encoder = best_model["data"]
    encoder[1].weight
    σ = best_model["σ"]
    encoder[1].weight ./= reshape(σ, 1, 20)  # rescale
    step = best_model["epoch"]

    data = joinpath(data_dir, test_data)
    data_test = load(joinpath(data, "test.jld2"))

    X_test = data_test["X"]

    (; cost) = trainer
    c_sum = 0
    c_max = 0
    c = 0
    model_builder = cbc_model
    try
        model_builder = grb_model
    catch
    end
    @showprogress for x in X_test
        c += 1
        solution_cost =
            cost(
                easy_problem(encoder(x.features); instance=x, model_builder=model_builder);
                instance=x,
            ) / x.city.nb_tasks
        c_max = max(solution_cost, c_max)
        c_sum += solution_cost
    end
    average_cost_per_task = c_sum / c

    @info "$log_dir -> $test_data" step c_max average_cost_per_task
    return Dict(
        "step" => step,
        "average_cost_per_task" => average_cost_per_task,
        "max_cost_per_task" => c_max,
    )
end

function evaluate_models(log_dirs, test_datasets, cost_only)
    for log_dir in log_dirs
        res = Dict()
        for test_data in test_datasets
            res[test_data] = evaluate_model(log_dir, test_data)
        end
        for test_data in cost_only
            res[test_data] = evaluate_cost(log_dir, test_data)
        end
        result_dir = joinpath(logdir, "results")
        if !isdir(result_dir)
            mkdir(result_dir)
        end
        jldsave(joinpath(result_dir, "$log_dir.jld2"); data=res)
        println("---")
    end
end

evaluate_models(log_dirs, test_datasets, cost_only)
