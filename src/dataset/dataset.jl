function compute_μ_σ(X)
    m = [mean(x.features, dims=2)[:, 1] for x in X] # mean per sample and feature
    μ = mean(m) # mean per feature
    σ = std(m)
    return μ, σ
end

function normalize_data!(X, μ, σ)
    for x in X
        for col in 1:size(x.features, 2)
            # @. x.features[:, col] = @views (x.features[:, col] - μ) / σ
            @. x.features[:, col] = @views (x.features[:, col]) / σ
        end
    end
end

# TODO : dataset config
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
    labeled=true,
    heuristic=true,
    city_kwargs
)
    if !isdir(dataset_folder)
        mkdir(dataset_folder)
    end

    nb_total_samples = nb_train_samples + nb_val_samples + nb_test_samples
    X, Y = generate_samples(nb_total_samples; heuristic=heuristic, labeled=labeled, city_kwargs)
    μ, σ = compute_μ_σ(X)
    normalize_data!(X, μ, σ)

    if nb_train_samples > 0
        train_slice = 1:nb_train_samples
        X_train = X[train_slice]
        train_file = joinpath(dataset_folder, "train.jld2")
        if labeled
            Y_train = Y[train_slice]
            jldsave(train_file, X=X_train, Y=Y_train)
        else
            jldsave(train_file, X=X_train)
        end
    end

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
