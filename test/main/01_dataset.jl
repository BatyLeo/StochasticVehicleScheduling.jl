using Random
using StochasticVehicleScheduling

Random.seed!(67)
dataset_folder = "data/100tasks10scenarios"
nb_tasks = 100
nb_scenarios = 10
city_kwargs = (; nb_tasks, nb_scenarios)

generate_dataset(dataset_folder, 50, 50, 50; city_kwargs);

# ---

using JLD2
using Graphs
using InferOpt
using StochasticVehicleScheduling.Training

data = load("data/data50_normalized/train.jld2");
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
