using Random
using StochasticVehicleScheduling

Random.seed!(67)

dataset_folder = "data/data50"
nb_tasks = 50
nb_scenarios = 10
city_kwargs = (; nb_tasks, nb_scenarios)

generate_dataset(dataset_folder, 50, 50; city_kwargs);
