# TODO : dataset config
function generate_samples(
    nb_samples::Integer;
    city_kwargs
)
    @info city_kwargs
    X = [Instance(create_random_city(; city_kwargs...)) for _ in 1:nb_samples]
    Y = [heuristic_solution(x; nb_it=10_000) for x in X]
    return X, Y
end

function generate_dataset(
    dataset_folder::String,
    nb_train_samples::Integer,
    nb_test_samples::Integer;
    city_kwargs
)
    if !isdir(dataset_folder)
        mkdir(dataset_folder)
    end
    X_train, Y_train = generate_samples(nb_train_samples; city_kwargs)
    train_file = joinpath(dataset_folder, "train.jld2")
    jldsave(train_file, X=X_train, Y=Y_train)
    X_test, Y_test = generate_samples(nb_test_samples; city_kwargs)
    test_file = joinpath(dataset_folder, "test.jld2")
    jldsave(test_file, X=X_test, Y=Y_test)
end

# """
#     generate_dataset(
#         dataset_size::Int;
#         nb_tasks::Int=20,
#         nb_scenarios::Int=100,
#         nb_it::Int=100,
#     )

# Returns a Vector containing `dataset_size` Instance, and a Vector containing `dataset_size`
#     Solution corresponding to instances.
#     Each instance contains `nb_tasks` tasks and `nb_scenarios` scenarios each,
#     Each solution is computed with `nb_it` iterations heuristic.
# """
# function generate_dataset(
#     dataset_size::Int;
#     nb_tasks::Int=20,
#     nb_scenarios::Int=100,
#     nb_it::Int=100,
# )
#     X = [Instance(create_random_city(nb_tasks=nb_tasks, nb_scenarios=nb_scenarios)) for _ in 1:dataset_size]
#     Y = [heuristic_solution(x; nb_it=nb_it) for x in X]
#     return (X=X, Y=Y)
# end

# """
#     save_dataset(dataset::Vector{Tuple{Instance, Solution}}, save_path::String)

# Save `dataset` at `dataset_path` location.
# """
# function save_dataset(X::Vector{<:Instance}, Y::Vector{Solution}, save_path::String)
#     jldsave(save_path, X=X, Y=Y)
#     return nothing
# end

# """
#     load_dataset(dataset_path::String)

# Load a dataset from `dataset_path` location.
# """
# function load_dataset(dataset_path::String)
#     f = jldopen(dataset_path)
#     return (X=f["X"], Y=f["Y"])
# end
