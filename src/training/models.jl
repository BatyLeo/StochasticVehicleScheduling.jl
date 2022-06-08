# abstract type AbstractModel end

# struct GeneralizedLinearModel{E, M} <: AbstractModel
#     encoder::E
#     maximizer::M
# end

# function GeneralizedLinearModel(;nb_features::Integer, model_builder)
#     encoder = Chain(Dense(nb_features => 1), vec)
#     maximizer(θ::AbstractVector; instance::Instance) = easy_problem(
#         θ; instance, model_builder=model_builder
#     )
#     return GeneralizedLinearModel(encoder, maximizer)
# end
struct Pipeline
    encoder
    maximizer
end

function (pipeline::Pipeline)(x)
    (; encoder, maximizer) = pipeline
    return maximizer(encoder(x.features), instance=x)
end

# ----------

struct Trainer
    data
    pipeline
    loss
    cost
    metrics
    opt
    logger
end

function Trainer(config, pipeline, loss; model_builder)
    (; data_dir, train_file, test_file) = config.data

    train_dir = joinpath(data_dir, train_file)
    dataset_train = load(train_dir);
    X_train, Y_train = dataset_train["X"], dataset_train["Y"];

    test_dir = joinpath(data_dir, test_file)
    dataset_test = load(test_dir);
    X_test, Y_test = dataset_test["X"], dataset_test["Y"];

    data = (train=SupervisedDataset(X_train, Y_train), test=SupervisedDataset(X_test, Y_test))

    cost(y; instance) = evaluate_solution(y, instance)
    metrics = [eval(Meta.parse(metric)) for metric in config.training.metrics]
    train_metrics = Tuple(metric() for metric in metrics)
    test_metrics = Tuple(metric() for metric in metrics)
    (; name, args) = config.training.optimizer
    opt = eval(Meta.parse(name))(args...)
    logger = TBLogger(joinpath(config.log_dir, config.experiment_name), min_level=Logging.Info)
    return Trainer(
        data,
        pipeline,
        loss,
        cost,
        (train=train_metrics, test=test_metrics),
        opt, logger
    )
end

function compute_metrics!(trainer::Trainer)
    Y_train_pred = [trainer.pipeline(x) for x in trainer.data.train.X]
    for (idx, metric) in enumerate(trainer.metrics.train)
        compute_value!(metric, trainer; train=true, Y_pred=Y_train_pred)
        log_last_measure!(metric, trainer.logger; train=true, step_increment=(idx==1 ? 1 : 0))
    end

    Y_test_pred = [trainer.pipeline(x) for x in trainer.data.test.X]
    for metric in trainer.metrics.test
        compute_value!(metric, trainer; train=false, Y_pred=Y_test_pred)
        log_last_measure!(metric, trainer.logger; train=false, step_increment=0)
    end
end

function train_loop!(trainer::Trainer, nb_epochs::Integer; show_progress=true)
    p = Progress(nb_epochs; enabled=show_progress)
    for _ in 1:nb_epochs
        compute_metrics!(trainer)
        Flux.train!(trainer.loss, Flux.params(trainer.pipeline.encoder), loss_data(trainer.data.train), trainer.opt)
        next!(p)
    end
end

# Specific constructors
function FenchelYoungGLM(config::NamedTuple; model_builder)
    nb_features = 20  # ! hardcoded
    encoder = Chain(Dense(nb_features => 1), vec)
    maximizer(θ::AbstractVector; instance) = easy_problem(
        θ; instance, model_builder=model_builder
    )
    pipeline = Pipeline(encoder, maximizer)
    ε = 0.1
    M = 5
    loss = FenchelYoungLoss(PerturbedNormal(maximizer; ε, M))
    pipeline_loss(x, y) = loss(encoder(x.features), y.value; instance=x)
    return Trainer(config, pipeline, pipeline_loss; model_builder)
end
