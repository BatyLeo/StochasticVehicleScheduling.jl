using StochasticVehicleScheduling
try
    using Gurobi
    @info "Gurobi loaded with success !"
catch
    @info "Gurobi not found, will use Cbc instead."
end

# List of settings for which we create a dataset
const settings = [
    (
        nb_tasks=25,
        nb_scenarios=10,
        labeled=true,
        heuristic=false,
        train=50,
        val=50,
        test=50,
    ),
    (nb_tasks=50, nb_scenarios=50, labeled=true, heuristic=true, train=50, val=50, test=50),
    (
        nb_tasks=100,
        nb_scenarios=50,
        labeled=true,
        heuristic=true,
        train=50,
        val=50,
        test=50,
    ),
    (
        nb_tasks=200,
        nb_scenarios=50,
        labeled=false,
        heuristic=false,
        train=0,
        val=0,
        test=50,
    ),
    (
        nb_tasks=300,
        nb_scenarios=10,
        labeled=false,
        heuristic=false,
        train=0,
        val=0,
        test=50,
    ),
    (
        nb_tasks=500,
        nb_scenarios=10,
        labeled=false,
        heuristic=false,
        train=0,
        val=0,
        test=50,
    ),
    (
        nb_tasks=750,
        nb_scenarios=10,
        labeled=false,
        heuristic=false,
        train=0,
        val=0,
        test=50,
    ),
    (
        nb_tasks=1000,
        nb_scenarios=10,
        labeled=false,
        heuristic=false,
        train=0,
        val=0,
        test=50,
    ),
]

function create_all_datasets(settings, data_dir="data")
    model_builder = cbc_model
    try
        model_builder = grb_model
    catch
    end

    if !isdir(data_dir)
        mkdir(data_dir)
    end

    for setting in settings
        (; nb_tasks, nb_scenarios, labeled, heuristic, train, val, test) = setting
        dataset_folder = joinpath(data_dir, "$(nb_tasks)tasks$(nb_scenarios)scenarios")

        generate_dataset(
            dataset_folder,
            train,
            val,
            test;
            labeled,
            heuristic,
            city_kwargs=(; nb_tasks, nb_scenarios),
            model_builder,
        )
    end
end

create_all_datasets(settings)

# ---

# Mixed
# using JLD2

# datasets = [
#     "25tasks10scenarios",
#     "50tasks50scenarios",
#     "75tasks50scenarios",
#     "100tasks50scenarios",
#     "200tasks50scenarios",
#     "300tasks10scenarios",
#     "400tasks10scenarios",
#     "500tasks10scenarios",
#     "750tasks10scenarios",
#     "1000tasks10scenarios",
# ];

# for setting in ["test"]
#     X = Instance[]
#     # Y = Solution[];
#     for dataset in datasets
#         data = load(joinpath("data", dataset, "$setting.jld2"))
#         X = cat(X, data["X"][1:5]; dims=1)
#         #Y = cat(Y, data["Y"][1:5], dims=1)
#     end
#     config = Dict(
#         "nb_samples" => Dict("train" => 0, "validation" => 0, "test" => 50),
#         "labeled" => false,
#         "heuristic" => false,
#         "city_kwargs" => "mixed",
#         "seed" => 67,
#     )
#     save_config(config, joinpath("data", "mixed", "info.yaml"))
#     slice = shuffle(1:length(X))
#     jldsave(joinpath("data", "mixed", "$setting.jld2"); X=X[slice])#, Y=Y[slice])
# end
