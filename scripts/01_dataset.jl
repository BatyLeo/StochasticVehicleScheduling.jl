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
