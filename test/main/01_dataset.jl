using StochasticVehicleScheduling
using InferOpt.Testing
using Random
using JLD2

Random.seed!(67)

nb_samples = 100
nb_tasks = 20
nb_scenarios = 10

X, Y = StochasticVehicleScheduling.generate_dataset(
    nb_samples; nb_tasks=nb_tasks, nb_scenarios=nb_scenarios, nb_it=10_000
)
X_train, X_test = train_test_split(X)
Y_train, Y_test = train_test_split(Y)
data = InferOptDataset(; X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)

save_path = "data/data.jld2"
jldsave(save_path, data=data)
