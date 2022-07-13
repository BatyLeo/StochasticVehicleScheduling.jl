using Random
using StochasticVehicleScheduling

Random.seed!(67)
nb_tasks = 200
nb_scenarios = 50
dataset_folder = "data/$(nb_tasks)tasks$(nb_scenarios)scenarios_uncentered"
city_kwargs = (; nb_tasks, nb_scenarios)

generate_dataset(dataset_folder, 50, 50, 50; labeled=true, heuristic=true, city_kwargs);

# ---

using JLD2

datasets = ["25tasks10scenarios_exact_uncentered", "50tasks50scenarios_uncentered", "100tasks50scenarios_uncentered"];

for setting in ["train", "validation", "test"]
    X = Instance[];
    Y = Solution[];
    for dataset in datasets
        data = load(joinpath("data", dataset, "$setting.jld2"))
        X = cat(X, data["X"][1:15], dims=1)
        Y = cat(Y, data["Y"][1:15], dims=1)
    end
    slice = shuffle(1:length(X))
    jldsave(joinpath("data", "mixed_uncentered", "$setting.jld2"), X=X[slice], Y=Y[slice])
end

# ---

using JLD2
using Graphs
using InferOpt
using StochasticVehicleScheduling.Training

data = load("data/1000tasks10scenarios/test.jld2");
X = data["X"];

μ, σ = compute_μ_σ(X)

μ
σ

config_file = "src/training/config.yaml"
trainer = Trainer(config_file);

x = X[1];

θ = trainer.pipeline.encoder(x.features)

trainer.pipeline.encoder[1].weight'

p = PerturbedComposition(PerturbedAdditive(trainer.pipeline.maximizer; ε=0.1, nb_samples=5, seed=0), trainer.cost)

p.perturbed(θ; instance=x)

sum(abs, θ) / length(θ)
