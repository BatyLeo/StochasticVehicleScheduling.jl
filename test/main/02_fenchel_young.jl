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

function main()
    Random.seed!(67);

    ## Dataset
    dataset_path = "data/data.jld2"
    data = load(dataset_path)["data"];

    ## GLM model and loss
    nb_features = 20
    encoder = Chain(Dense(nb_features => 1), vec)
    loss = FenchelYoungLoss(PerturbedNormal(easy_problem; ε=0.1, M=5))

    ## Training setup
    pipeline(x) = easy_problem(encoder(x.features), instance=x);
    pipeline_loss(x, y) = loss(encoder(x.features), y.value; instance=x)
    cost(y; instance) = evaluate_solution(y, instance)
    metrics = Dict("loss" => Loss)#, "cost gap" => CostGap)
    extra_info = (; cost)

    trainer = InferOptTrainer(;
        metrics_dict=metrics,
        pipeline=pipeline,
        loss=pipeline_loss,
        extra_info=extra_info,
    )
    @info "" typeof(trainer)

    nb_epochs = 20
    opt = ADAM()
    logger = TBLogger("tensorboard_logs/test_run", min_level=Logging.Info)

    # Training loop
    @showprogress for _ in 1:nb_epochs
        compute_metrics!(trainer, data; logger=logger)
        (;X, Y) = data.train
        Flux.train!(trainer.loss, Flux.params(encoder), zip(X, Y), opt)
    end

    # Results
    plot_perf(trainer; lineplot_function=lineplot)
    test_perf(trainer)
end

@profview main()
