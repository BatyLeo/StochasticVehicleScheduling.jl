using Random
using StochasticVehicleScheduling
using StochasticVehicleScheduling.Training
using UnicodePlots

Random.seed!(67);
config_file = "src/training/config2.yaml"
trainer = Trainer(config_file);
train_loop!(trainer, 10, show_progress=true)
plot_perf(trainer; lineplot_function=lineplot)
