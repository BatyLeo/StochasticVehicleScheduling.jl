using Random
using StochasticVehicleScheduling
using StochasticVehicleScheduling.Training
using UnicodePlots

Random.seed!(67);
config_file = "src/training/config2.yaml"
trainer = Trainer(config_file);
train_loop!(trainer, 10, show_progress=true)
plot_perf(trainer; lineplot_function=lineplot)

using JLD2, Flux
dataset_train = load("data/data50/train.jld2");
X_train, Y_train = dataset_train["X"], dataset_train["Y"];
data = Flux.DataLoader((X_train, ); batchsize=1);
dataxy = Flux.DataLoader((X_train, Y_train); batchsize=1);

for d in data
    @info typeof(d)
end
