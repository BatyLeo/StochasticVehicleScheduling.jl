using Random
using StochasticVehicleScheduling

Random.seed!(67)


dataset_folder = "data/data50"
nb_tasks = 50
nb_scenarios = 10
city_kwargs = (; nb_tasks, nb_scenarios)

StochasticVehicleScheduling.generate_dataset(
    dataset_folder, 50, 50; city_kwargs
);
# X_train, X_test = train_test_split(X);
# Y_train, Y_test = train_test_split(Y);
# data = InferOptDataset(; X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test);

# save_path = "data/data50.jld2"
# jldsave(save_path, data=data)
