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

function main(data_file)
    Random.seed!(67);

    ## Dataset
    data = load("data/$data_file.jld2")["data"];

    ## GLM model and loss
    nb_features = 20
    ε = 1000
    encoder = Chain(Dense(nb_features => 1), vec)
    #encoder = load("logs/FY_data100_20_eps0.1/model.jld2")["data"]
    cost(y; instance) = evaluate_solution(y, instance)
    loss = PerturbedCost(PerturbedNormal(easy_problem; ε=ε, M=5), cost)
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

    nb_epochs = 300
    opt = ADAM()
    log_dir = "logs/perturbed_$(data_file)_eps$(ε)_300epochs"
    logger = TBLogger(log_dir, min_level=Logging.Info)

    # Training loop
    @showprogress for _ in 1:nb_epochs
        compute_metrics!(trainer, data; logger=logger)
        (;X) = data.train
        Flux.train!(trainer.loss, Flux.params(encoder), zip(X), opt)
    end

    # Results
    plot_perf(trainer; lineplot_function=lineplot)
    test_perf(trainer)

    jldsave("$log_dir/model.jld2", data=encoder)
end

main("data100_20")

#model = load("data/model2.jld2")["data"];
#first(model).weight[1, :]
