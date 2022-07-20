function compute_μ_σ(X)
    nb_arcs = 0
    nb_features = size(X[1].features, 1)
    μ = zeros(nb_features)
    σ = zeros(nb_features)
    maxi = zeros(nb_features)

    for x in X
        features = x.features
        μ .+= sum(features, dims=2)
        nb_arcs += ne(x.graph)
        maxi = max.(maxi, maximum(features, dims=2)[:, 1])
    end
    μ ./= nb_arcs

    for x in X
        features = x.features
        σ .+= sum((features .- μ) .^ 2, dims=2)
    end
    σ ./= nb_arcs
    σ = sqrt.(σ)

    return μ, σ, maxi[:, 1]
end

"""
Standardize data.
"""
function normalize_data!(X, μ, σ)
    for x in X
        for col in 1:size(x.features, 2)
            @. x.features[:, col] = @views (x.features[:, col] - μ) / σ
        end
    end
end

"""
Reduce data, without centering it.
"""
function reduce_data!(X, σ)
    for x in X
        for col in 1:size(x.features, 2)
            @. x.features[:, col] = @views (x.features[:, col]) / σ
        end
    end
end

function generate_samples(nb_samples::Integer; heuristic=true, labeled=true, city_kwargs)
    @info city_kwargs
    X = [Instance(create_random_city(; city_kwargs...)) for _ in 1:nb_samples]
    if !labeled
        Y = Solution[]
    elseif heuristic
        Y = [heuristic_solution(x; nb_it=10_000) for x in X]
        # Y = [solve_deterministic_VSP(x; include_delays=true)[2] for x in X]
    else
        Y = [solve_scenarios(x; model_builder=grb_model)[2] for x in X]
    end
    return X, Y
end

function generate_dataset(
    dataset_folder::String,
    nb_train_samples::Integer,
    nb_val_samples::Integer,
    nb_test_samples::Integer;
    random_seed=67,
    labeled=true,
    heuristic=true,
    city_kwargs
)
    config = Dict(
        "nb_samples" => Dict(
            "train" => nb_train_samples,
            "validation" => nb_val_samples,
            "test" => nb_test_samples,
        ),
        "labeled" => labeled,
        "heuristic" => heuristic,
        "city_kwargs" => city_kwargs,
        "seed" => random_seed,
    )

    if !isdir(dataset_folder)
        mkdir(dataset_folder)
    end

    nb_total_samples = nb_train_samples + nb_val_samples + nb_test_samples

    # Fix the seed and generate all the samples
    Random.seed!(67)
    X, Y = generate_samples(nb_total_samples; heuristic=heuristic, labeled=labeled, city_kwargs)

    if nb_train_samples > 0
        train_slice = 1:nb_train_samples
        X_train = X[train_slice]
        μ, σ, maxi = compute_μ_σ(X)
        config["μ_train"] = μ
        config["σ_train"] = σ
        config["max"] = maxi
        train_file = joinpath(dataset_folder, "train.jld2")
        if labeled
            Y_train = Y[train_slice]
            jldsave(train_file, X=X_train, Y=Y_train)
        else
            jldsave(train_file, X=X_train)
        end
    end

    save_config(config, joinpath(dataset_folder, "info.yaml"))

    if nb_val_samples > 0
        val_slice = nb_train_samples+1:nb_train_samples+nb_val_samples
        validation_file = joinpath(dataset_folder, "validation.jld2")
        X_val = X[val_slice]
        if labeled
            Y_val = Y[val_slice]
            jldsave(validation_file, X=X_val, Y=Y_val)
        else
            jldsave(validation_file, X=X_val)
        end
    end

    if nb_test_samples > 0
        test_slice = nb_train_samples+nb_val_samples+1:nb_total_samples

        X_test = X[test_slice]
        test_file = joinpath(dataset_folder, "test.jld2")
        if labeled
            Y_test = Y[test_slice]
            jldsave(test_file, X=X_test, Y=Y_test)
        else
            jldsave(test_file, X=X_test)
        end
    end
end
