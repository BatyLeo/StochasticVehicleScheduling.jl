using StochasticVehicleScheduling
using JLD2
using Flux

dataset_path = joinpath("data", "50tasks50scenarios", "test.jld2");
data = load(dataset_path);
X_test = data["X"];
Y_test = data["Y"];

model_path = joinpath("final_experiments", "imitation_25tasks10scenarios_1", "best.jld2");
model = load(model_path);
encoder = model["data"]
encoder[1].weight'
Flux.params(encoder)[1]'

θ_pred = [encoder(x.features) for x in X_test];
Y_pred = [easy_problem(encoder(x.features); instance=x, model_builder=grb_model) for x in X_test];

model

x, y_pred = X_test[1], Y_pred[1];

sol = Solution(y_pred, x);
get_routes(sol)
evaluate_solution(sol, x)

basic = basic_solution(x)
get_routes(basic)
evaluate_solution(basic, x)

θ_pred[1]
x.features
