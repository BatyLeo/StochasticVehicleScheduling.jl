"""
    Pipeline

InferOpt pipeline container an encoder and a maximizer
"""
struct Pipeline{E,M}
    encoder::E
    maximizer::M
end

function (pipeline::Pipeline)(x)
    (; encoder, maximizer) = pipeline
    return maximizer(encoder(x.features); instance=x)
end

# ----------

"""
    Trainer

Main structure used for training an InferOpt model.
"""
struct Trainer{D,P,L,C,M,O,LL,F}
    data::D
    pipeline::P
    loss::L
    cost::C
    metrics::M
    opt::O
    logger::LL
    log_every_n_epochs::Int
    save_every_n_epochs::Int
    nb_epochs::Int
    normalization_factor::F
end

function Trainer(config_file::String; create_logger=true)
    config = read_config(config_file)
    return Trainer(config; create_logger=true)
end

function Trainer(config; create_logger=true)
    if create_logger
        logger = TBLogger(
            joinpath(config.train.log_dir, config.train.tag); min_level=Logging.Info
        )
        save_config(config, joinpath(logger.logdir, "config.yaml"))
    else
        logger = nothing
    end

    # Build model
    model = eval(Meta.parse(config.model.name))
    pipeline, loss, cost, dataset = model(; config.model.args...)

    # Load data and create dataset
    (; data_dir, train_file, validation_file) = config.data

    train_dir = joinpath(data_dir, train_file)
    dataset_train = load(train_dir)
    X_train, Y_train = dataset_train["X"], dataset_train["Y"] # TODO: remove hardcoded stuff

    val_dir = joinpath(data_dir, validation_file)
    dataset_val = load(val_dir)
    X_val, Y_val = dataset_val["X"], dataset_val["Y"] # TODO: remove hardcoded stuff

    # Normalize data by reducing by its variance
    data_info = read_config(joinpath(data_dir, "info.yaml"))
    σ = data_info.σ_train
    reduce_data!(X_train, σ)
    reduce_data!(X_val, σ)

    data_train = dataset(X_train, Y_train)
    data_val = dataset(X_val, Y_val)
    loader = build_loader(data_train, config.train.batchsize)
    data = (loader=loader, train=data_train, validation=data_val)

    # Metrics
    metrics = [
        eval(Meta.parse(metric)) for metric in config.train.metrics.train_and_validation
    ]
    train_metric_list = if isnothing(config.train.metrics.train)
        []
    else
        [eval(Meta.parse(metric)) for metric in config.train.metrics.train]
    end
    val_metric_list = if isnothing(config.train.metrics.validation)
        []
    else
        [eval(Meta.parse(metric)) for metric in config.train.metrics.validation]
    end

    train_metrics = Tuple(metric() for metric in cat(metrics, train_metric_list; dims=1))
    validation_metrics = Tuple(metric() for metric in cat(metrics, val_metric_list; dims=1))

    # Optimizer
    (; name, args) = config.train.optimizer
    if isnothing(args)
        opt = eval(Meta.parse(name))()
    else
        opt = eval(Meta.parse(name))(args...)
    end

    (; log_every_n_epochs, save_every_n_epochs, nb_epochs) = config.train

    return Trainer(
        data,
        pipeline,
        loss,
        cost,
        (train=train_metrics, validation=validation_metrics),
        opt,
        logger,
        log_every_n_epochs,
        save_every_n_epochs,
        nb_epochs,
        σ,
    )
end

function compute_metrics!(trainer::Trainer, epoch::Integer)
    if epoch % trainer.log_every_n_epochs != 0
        return nothing
    end

    # else
    Y_train_pred = [trainer.pipeline(x) for x in trainer.data.train.X]
    for (idx, metric) in enumerate(trainer.metrics.train)
        compute_value!(metric, trainer; train=true, Y_pred=Y_train_pred, epoch=epoch)
        log_last_measure!(
            metric,
            trainer.logger;
            train=true,
            step_increment=(idx == 1 ? epoch - trainer.logger.global_step : 0),
        )
    end

    Y_val_pred = [trainer.pipeline(x) for x in trainer.data.validation.X]
    for metric in trainer.metrics.validation
        compute_value!(metric, trainer; train=false, Y_pred=Y_val_pred, epoch=epoch)
        log_last_measure!(metric, trainer.logger; train=false, step_increment=0)
    end
    return nothing
end

function save_model(trainer::Trainer, epoch::Integer; best=false)
    if best
        jldsave(
            "$(trainer.logger.logdir)/best.jld2";
            data=trainer.pipeline.encoder,
            epoch=epoch,
            σ=trainer.normalization_factor,
        )
        return nothing
    end

    # else
    if epoch % trainer.save_every_n_epochs != 0
        return nothing
    end
    # else
    return jldsave(
        "$(trainer.logger.logdir)/model_$epoch.jld2";
        data=trainer.pipeline.encoder,
        σ=trainer.normalization_factor,
    )
end

function my_custom_train!(loss, ps, data, opt)
    local training_loss
    #grad_sum = 0.0
    for batch in data
        gs = gradient(ps) do
            training_loss = loss(batch...)
            return training_loss
        end
        # n = 0
        # total = 0.0
        # for p in ps
        #     g = gs[p]
        #     total += sum(abs, g)
        #     n += length(g)
        # end
        # grad_sum += total / n
        Flux.update!(opt, ps, gs)
    end
    #@info grad_sum / length(data)
    return nothing
end

function train_loop!(trainer::Trainer; show_progress=true)
    (; nb_epochs, loss, pipeline, opt, data) = trainer
    p = Progress(nb_epochs; enabled=show_progress)
    compute_metrics!(trainer, 0)
    for n in 1:nb_epochs
        my_custom_train!(loss, Flux.params(pipeline.encoder), data.loader, opt)
        #Flux.train!(trainer.loss, Flux.params(trainer.pipeline.encoder), loader(trainer.data.train), trainer.opt)
        compute_metrics!(trainer, n)
        save_model(trainer, n)
        next!(p)
    end
end

# Specific constructors

# --

# function normalizing(x::Vector)
#     return x / LinearAlgebra.norm(x)
# end
