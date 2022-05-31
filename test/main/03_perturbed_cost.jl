using Flux
using InferOpt
using InferOpt.Testing
using JLD2
using Logging
using ProgressMeter
using Random
using StochasticVehicleScheduling
using TensorBoardLogger
using UnicodePlots

function main(dataset_path="data/data.jld2")
    Random.seed!(67);

    ## Dataset
    data = load(dataset_path)["data"];

    ## GLM model and loss
    nb_features = 20
    encoder = Chain(Dense(nb_features => 1), vec)
    cost(y; instance) = evaluate_solution(y, instance)
    loss = PerturbedCost(PerturbedNormal(easy_problem; ε=1000, M=5), cost)
    penalty() = 1000 * sum(abs, first(encoder).weight)

    ## Training setup
    pipeline(x) = easy_problem(encoder(x.features), instance=x);
    pipeline_loss(x) = loss(encoder(x.features); instance=x) + penalty()
    metrics = Dict("loss" => Loss, "cost gap" => CostGap)
    extra_info = (; cost)

    trainer = InferOptTrainer(;
        metrics_dict=metrics,
        pipeline=pipeline,
        loss=pipeline_loss,
        extra_info=extra_info,
    )
    @info "" typeof(trainer)

    nb_epochs = 100
    opt = ADAM()
    logger = TBLogger("tensorboard_logs/perturbed_cost50", min_level=Logging.Info)

    # Training loop
    @showprogress for _ in 1:nb_epochs
        compute_metrics!(trainer, data; logger=logger)
        (;X) = data.train
        Flux.train!(trainer.loss, Flux.params(encoder), zip(X), opt)
    end

    # Results
    plot_perf(trainer; lineplot_function=lineplot)
    test_perf(trainer)

    jldsave("data/model2.jld2", data=encoder)
end

main("data/data50.jld2")

model = load("data/model2.jld2")["data"];
first(model).weight[1, :]
