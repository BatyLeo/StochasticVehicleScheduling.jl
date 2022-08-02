
function PerturbedGLM(; nb_features, ε, M, model_builder::String, seed=nothing)
    encoder = Chain(Dense(nb_features => 1; bias=false), vec)
    function maximizer(θ::AbstractVector; instance)
        return easy_problem(θ; instance, model_builder=eval(Meta.parse(model_builder)))
    end
    pipeline = Pipeline(encoder, maximizer)

    cost(y; instance) = evaluate_solution(y, instance)

    loss = ProbabilisticComposition(
        PerturbedAdditive(maximizer; ε=ε, nb_samples=M, seed=seed), cost
    )
    pipeline_loss(X) = mean(loss(encoder(x.features); instance=x) for x in X)

    return pipeline, pipeline_loss, cost, ExperienceDataset
end

function PerturbedMultiplicativeGLM(;
    nb_features, ε, M, model_builder::String, seed=nothing
)
    encoder = Chain(Dense(nb_features => 1; bias=false), vec)
    function maximizer(θ::AbstractVector; instance)
        return easy_problem(θ; instance, model_builder=eval(Meta.parse(model_builder)))
    end
    pipeline = Pipeline(encoder, maximizer)

    cost(y; instance) = evaluate_solution(y, instance)

    loss = ProbabilisticComposition(
        PerturbedMultiplicative(maximizer; ε=ε, nb_samples=M, seed=seed), cost
    )
    pipeline_loss(X) = mean(loss(encoder(x.features); instance=x) for x in X)

    return pipeline, pipeline_loss, cost, ExperienceDataset
end

# --

function build_θ(fg)
    features = node_feature(fg)
    return [features[e[1], e[2]] for (i, e) in edges(fg)]
end

function GNN_imitation(; nb_features, ε, M, model_builder::String)
    encoder = Chain(
        GCNConv(nb_features => 20),
        GraphParallel(; node_layer=Dense(20, nb_vertices)),
        build_θ,
    )

    function maximizer(θ::AbstractVector; instance)
        return easy_problem(θ; instance, model_builder=eval(Meta.parse(model_builder)))
    end
    pipeline = Pipeline(encoder, maximizer)

    loss = FenchelYoungLoss(PerturbedNormal(maximizer; ε, M))
    function pipeline_loss(X, Y)
        return mean(loss(encoder(x.features), y.value; instance=x) for (x, y) in zip(X, Y))
    end

    cost(y; instance) = evaluate_solution(y, instance)
    return pipeline, pipeline_loss, cost, SupervisedDataset
end
