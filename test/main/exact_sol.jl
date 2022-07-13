using JLD2
using Graphs
using InferOpt
using StochasticVehicleScheduling
using StochasticVehicleScheduling.Training

data = load("data/50tasks10scenarios/train.jld2");
X = data["X"];
Y = data["Y"];

index = 1
x = X[index];
y = Y[index];

@time solve_scenarios(x; model_builder=grb_model)

evaluate_solution(y, x)
