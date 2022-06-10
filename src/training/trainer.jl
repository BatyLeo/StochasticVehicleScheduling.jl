struct Pipeline{E, M}
    encoder::E
    maximizer::M
end

function (pipeline::Pipeline)(x)
    (; encoder, maximizer) = pipeline
    return maximizer(encoder(x.features), instance=x)
end

# ----------

struct Trainer{D, P, L, C, M, O, LL}
    data::D
    pipeline::P
    loss::L
    cost::C
    metrics::M
    opt::O
    logger::LL
    log_every_n_epochs::Int
    save_every_n_epochs::Int
end

function Trainer(config_file)#; model_builder)
    config = read_config(config_file);
    logger = TBLogger(joinpath(config.train.log_dir, config.train.tag), min_level=Logging.Info)
    save_config(config, joinpath(logger.logdir, "config.yaml"))

    # Load data and create dataset
    (; data_dir, train_file, test_file, dataset_type) = config.data

    train_dir = joinpath(data_dir, train_file)
    dataset_train = load(train_dir);
    X_train, Y_train = dataset_train["X"], dataset_train["Y"]; # TODO: remove hardcoded stuff

    test_dir = joinpath(data_dir, test_file)
    dataset_test = load(test_dir);
    X_test, Y_test = dataset_test["X"], dataset_test["Y"]; # TODO: remove hardcoded stuff

    dataset = eval(Meta.parse(dataset_type))
    data_train = dataset(X_train, Y_train)
    data_test = dataset(X_test, Y_test)
    loader = build_loader(data_train, config.train.batchsize)
    data = (loader=loader, train=data_train, test=data_test)

    # Build model
    model = eval(Meta.parse(config.model.name))
    pipeline, loss, cost = model(; config.model.args...)

    # Metrics
    metrics = [eval(Meta.parse(metric)) for metric in config.train.metrics]
    train_metrics = Tuple(metric() for metric in metrics)
    test_metrics = Tuple(metric() for metric in metrics)

    # Optimizer
    (; name, args) = config.train.optimizer
    opt = eval(Meta.parse(name))(args...)

    (; log_every_n_epochs, save_every_n_epochs) = config.train

    return Trainer(
        data,
        pipeline,
        loss,
        cost,
        (train=train_metrics, test=test_metrics),
        opt,
        logger,
        log_every_n_epochs,
        save_every_n_epochs,
    )
end

function compute_metrics!(trainer::Trainer, n::Integer)
    if n % trainer.log_every_n_epochs != 0
        return nothing
    end
    # else
    Y_train_pred = [trainer.pipeline(x) for x in trainer.data.train.X]
    for (idx, metric) in enumerate(trainer.metrics.train)
        compute_value!(metric, trainer; train=true, Y_pred=Y_train_pred)
        log_last_measure!(metric, trainer.logger; train=true, step_increment=(idx==1 ? n-trainer.logger.global_step : 0))
    end

    Y_test_pred = [trainer.pipeline(x) for x in trainer.data.test.X]
    for metric in trainer.metrics.test
        compute_value!(metric, trainer; train=false, Y_pred=Y_test_pred)
        log_last_measure!(metric, trainer.logger; train=false, step_increment=0)
    end
    return nothing
end

function save_model(trainer::Trainer, n::Integer)
    if n % trainer.save_every_n_epochs != 0
        return nothing
    end
    # else
    jldsave("$(trainer.logger.logdir)/model_$n.jld2", data=trainer.pipeline.encoder)
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

function train_loop!(trainer::Trainer, nb_epochs::Integer; show_progress=true)
    p = Progress(nb_epochs; enabled=show_progress)
    compute_metrics!(trainer, 0)
    for n in 1:nb_epochs
        my_custom_train!(trainer.loss, Flux.params(trainer.pipeline.encoder), trainer.data.loader, trainer.opt)
        #Flux.train!(trainer.loss, Flux.params(trainer.pipeline.encoder), loader(trainer.data.train), trainer.opt)
        compute_metrics!(trainer, n)
        save_model(trainer, n)
        next!(p)
    end
end

# Specific constructors
function FenchelYoungGLM(; nb_features, ε, M, model_builder::String)
    encoder = Chain(Dense(nb_features => 1), vec)
    maximizer(θ::AbstractVector; instance) = easy_problem(
        θ; instance, model_builder=eval(Meta.parse(model_builder))
    )
    pipeline = Pipeline(encoder, maximizer)

    loss = FenchelYoungLoss(PerturbedNormal(maximizer; ε, M))
    pipeline_loss(X, Y) = mean(loss(encoder(x.features), y.value; instance=x) for (x, y) in zip(X, Y))

    cost(y; instance) = evaluate_solution(y, instance)
    return pipeline, pipeline_loss, cost
end

function PerturbedGLM(; nb_features, ε, M, model_builder::String)
    encoder = Chain(Dense(nb_features => 1), vec)
    maximizer(θ::AbstractVector; instance) = easy_problem(
        θ; instance, model_builder=eval(Meta.parse(model_builder))
    )
    pipeline = Pipeline(encoder, maximizer)

    cost(y; instance) = evaluate_solution(y, instance)

    loss = PerturbedCost(PerturbedNormal(maximizer; ε=ε, M=M), cost)
    pipeline_loss(X) = mean(loss(encoder(x.features); instance=x) for x in X)

    return pipeline, pipeline_loss, cost
end
