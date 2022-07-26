using Random
using StochasticVehicleScheduling

nb_tasks = 75
nb_scenarios = 50

city_kwargs = (; nb_tasks, nb_scenarios)
dataset_folder = "data/$(nb_tasks)tasks$(nb_scenarios)scenarios"

generate_dataset(dataset_folder, 0, 0, 50; labeled=true, heuristic=true, city_kwargs);

# ---

# Mixed
using JLD2

datasets = [
    "25tasks10scenarios",
    "50tasks50scenarios",
    "75tasks50scenarios",
    "100tasks50scenarios",
    "200tasks50scenarios",
    "300tasks10scenarios",
    "400tasks10scenarios",
    "500tasks10scenarios",
    "750tasks10scenarios",
    "1000tasks10scenarios",
];

for setting in ["test"]
    X = Instance[];
    # Y = Solution[];
    for dataset in datasets
        data = load(joinpath("data", dataset, "$setting.jld2"))
        X = cat(X, data["X"][1:5], dims=1)
        #Y = cat(Y, data["Y"][1:5], dims=1)
    end
    config = Dict(
        "nb_samples" => Dict(
            "train" => 0,
            "validation" => 0,
            "test" => 50,
        ),
        "labeled" => false,
        "heuristic" => false,
        "city_kwargs" => "mixed",
        "seed" => 67,
    )
    save_config(config, joinpath("data", "mixed", "info.yaml"))
    slice = shuffle(1:length(X))
    jldsave(joinpath("data", "mixed", "$setting.jld2"), X=X[slice])#, Y=Y[slice])
end

# ---

# Random stuff tests

# using JLD2
# using Graphs
# using InferOpt
# using StochasticVehicleScheduling.Training

# data = load("data/1000tasks10scenarios/test.jld2");
# X = data["X"];

# μ, σ = compute_μ_σ(X)

# μ
# σ

# config_file = "src/training/config.yaml"
# trainer = Trainer(config_file);

# x = X[1];

# θ = trainer.pipeline.encoder(x.features)

# trainer.pipeline.encoder[1].weight'

# p = PerturbedComposition(PerturbedAdditive(trainer.pipeline.maximizer; ε=0.1, nb_samples=5, seed=0), trainer.cost)

# p.perturbed(θ; instance=x)

# sum(abs, θ) / length(θ)
