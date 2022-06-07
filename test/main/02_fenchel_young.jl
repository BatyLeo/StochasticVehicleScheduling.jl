using CUDA
using Flux
using Gurobi
using InferOpt
using InferOpt.Testing
using JLD2
using JuMP
using Logging
using ProgressMeter
using Random
using StochasticVehicleScheduling
using TensorBoardLogger
using UnicodePlots

const GRB_ENV = Gurobi.Env()

function grb_model()
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    return model
end

function main(data_file)
    Random.seed!(67);

    ## Dataset
    data = load("data/$data_file.jld2")["data"]

    ## GLM model and loss
    nb_features = 20
    encoder = Chain(Dense(nb_features => 1), vec)
    maximizer(θ::AbstractVector; instance::Instance) = easy_problem(θ; instance, model_builder=grb_model)
    ε = 0.1
    loss = FenchelYoungLoss(PerturbedNormal(maximizer; ε=ε, M=5))

    ## Training setup
    pipeline(x) = maximizer(encoder(x.features), instance=x)
    pipeline_loss(x, y) = loss(encoder(x.features), y.value; instance=x)
    cost(y; instance) = evaluate_solution(y, instance)
    metrics = Dict("loss" => Loss, "cost gap" => CostGap)
    extra_info = (; cost)

    trainer = InferOptTrainer(;
        metrics_dict=metrics,
        pipeline=pipeline,
        loss=pipeline_loss,
        extra_info=extra_info,
    )
    @info "Trainer" typeof(trainer)

    nb_epochs = 10
    opt = ADAM()
    log_dir = "tensorboard_logs/FY_$(data_file)_eps$ε"
    logger = TBLogger(log_dir, min_level=Logging.Info)

    # Training loop
    @showprogress for _ in 1:nb_epochs
        compute_metrics!(trainer, data; logger=logger)
        (;X, Y) = data.train
        Flux.train!(trainer.loss, Flux.params(encoder), zip(X, Y), opt)
    end

    # Results
    plot_perf(trainer; lineplot_function=lineplot)
    test_perf(trainer)

    #jldsave("$log_dir/model.jld2", data=encoder)
end

@profview main("data50")
