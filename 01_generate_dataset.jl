include("00_config.jl")

using Gurobi: Gurobi

# Config
model_builder = grb_model # mip solver
num_iterations = 10_000 # local search iterations
test_N = [200, 300, 500, 750, 1000] # test set number of tasks

b = StochasticVehicleSchedulingBenchmark(; nb_tasks=25, nb_scenarios=10)
b_50 = StochasticVehicleSchedulingBenchmark(; nb_tasks=50, nb_scenarios=50)
b_100 = StochasticVehicleSchedulingBenchmark(; nb_tasks=100, nb_scenarios=50)

dataset_25 = generate_dataset(b, 150; algorithm=compact_mip, model_builder, silent=false);
dataset_50 = generate_dataset(b_50, 150; algorithm=local_search, num_iterations);
dataset_100 = generate_dataset(b_100, 150; algorithm=local_search, num_iterations);

train_set_25, val_set_25, test_set_25 = splitobs(dataset_25; at=(50, 50));
train_set_50, val_set_50, test_set_50 = splitobs(dataset_50; at=(50, 50));
train_set_100, val_set_100, test_set_100 = splitobs(dataset_100; at=(50, 50));

# Feature normalization
dt = StatsBase.fit(StatsBase.ZScoreTransform, train_set_25; center=false, scale=true);
StatsBase.transform!(dt, dataset_25)
StatsBase.transform!(dt, dataset_50)
StatsBase.transform!(dt, dataset_100)

JLD2.jldsave(dataset_path; dataset_25, dataset_50, dataset_100, dt)

# Test datasets with larger instances if needed (can be quite heavy in disk space)
# test_datasets = Dict(
#     map(test_N) do N
#         b_N = StochasticVehicleSchedulingBenchmark(; nb_tasks=N, nb_scenarios=50)
#         dataset_N = generate_dataset(b_N, 50; compute_solutions=false, store_city=false)
#         StatsBase.transform!(dt, dataset_N)
#         return N => dataset_N
#     end,
# );
# JLD2.jldsave(test_dataset_path; dataset_25, dataset_50, dataset_100, dt)
