## Imports useful packages
using InferOptModelZoo.VSP
using InferOpt
using InferOptExperimental
using Random, Test
using Flux
using UnicodePlots
using Statistics: mean
using ProgressMeter
Random.seed!(1);

## Dataset creation

nb_samples = 100
split_ratio = 0.5
nb_tasks = 20
nb_scenarios = 10

function train_test_split(X::AbstractVector, Y::AbstractVector, split_ratio::Real)
    @assert length(X) == length(Y) "X and Y have different lengths"
    nb_training_samples = length(X) - trunc(Int, length(X) * split_ratio)
    X_train, Y_train = X[1:nb_training_samples], Y[1:nb_training_samples]
    X_test, Y_test = X[nb_training_samples+1:end], Y[nb_training_samples+1:end]
    return (X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
end

X, Y = generate_dataset(nb_samples; nb_tasks=nb_tasks, nb_scenarios=nb_scenarios, nb_it=10_000);
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, split_ratio);
nb_features = size(X_train[1].features, 1);
data_train, data_test = zip(X_train, Y_train), zip(X_test, Y_test);

## Initialization

# Initialize the GLM predictor:
model = Chain(Dense(nb_features => 1), vec)

pipeline(x) = easy_problem(model(x.features), instance=x);

initial_pred = [pipeline(x) for x in X_test]
initial_mean_obj = mean(evaluate_solution(y, x) for (x, y) in zip(X_test, initial_pred))
ground_truth_obj = [evaluate_solution(y, x) for (x, y) in data_test]
ground_truth_mean_obj = mean(ground_truth_obj)
@info "Ground truth" ground_truth_mean_obj
@info "Initial" initial_mean_obj
initial_obj_gap = initial_mean_obj - ground_truth_mean_obj
@info "Difference" initial_obj_gap

## Loss function
ε = 0.1
M = 5
# loss = FenchelYoungLoss(Perturbed(easy_problem; ε=ε, M=M))
loss = FenchelYoungLoss(
    FrankWolfeRegularizedPrediction(;
        Ω=InferOpt.half_square_norm,
        ∇Ω=identity,
        linear_maximizer=easy_problem,
    )
)

flux_loss(x, y) = loss(model(x.features), y.value; linear_maximizer_kwargs=(;instance=x))
opt = ADAM();
## Training loop

# We train our model for 200 epochs
nb_epochs = 100
hamming_distance(x::AbstractVector, y::AbstractVector) = sum(x[i] != y[i] for i in eachindex(x))
training_losses, test_losses = Float64[], Float64[]
Δ_objective_history, hamming_distances = Float64[], Float64[]
@showprogress for epoch in 1:nb_epochs
    l = mean(flux_loss(x, y) for (x, y) in data_train)
    l_test = mean(flux_loss(x, y) for (x, y) in data_test)
    Y_pred = [easy_problem(model(x.features); instance=x) for x in  X_test]
    values = [evaluate_solution(y, x) for (x, y) in zip(X_test, Y_pred)]
    V = mean((v_pred - v) / abs(v) for (v_pred, v) in zip(values, ground_truth_obj))
    H = mean(hamming_distance(y_pred, y.value) for (y_pred, y) in zip(Y_pred, Y_test))
    push!(training_losses, l)
    push!(test_losses, l_test)
    push!(Δ_objective_history, V)
    push!(hamming_distances, H)

    @info "Epoch $epoch" l l_test V

    Flux.train!(flux_loss, Flux.params(model), data_train, opt)
end

## Results

println(lineplot(training_losses, title="Training loss"))
println(lineplot(test_losses, title="Test loss"))
@info "Initial/final training losses" training_losses[1] training_losses[end]
@info "Initial/final test loss" test_losses[1] test_losses[end]

println(lineplot(Δ_objective_history, title="Objective difference"))
@info "Initial objective difference" Δ_objective_history[1]
@info "Final objective difference" Δ_objective_history[end]

println(lineplot(hamming_distances, title="Hamming distance"))
@info "Initial/final average hamming distance" hamming_distances[1] hamming_distances[end]
