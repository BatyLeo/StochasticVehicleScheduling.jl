# # End-to-end learning pipeline using `InferOpt`

#=
Here is the end-to-end learning pipeline we will build (see [here](https://axelparmentier.github.io/InferOpt.jl/dev/math/#Structured-learning-pipeline) for more details):
```math
\xrightarrow[\text{Instance}]{X}
\fbox{Encoder $\varphi_w$}
\xrightarrow[\text{Cost vector}]{\theta \in \mathbb{R}^{d(x)}}
\fbox{Easy problem}
\xrightarrow[\text{Solution}]{y \in \mathcal{Y}(x)}
```

- Let ``x`` an input instance. For each arc ``a\in A``, we compute a features vector ``x_a\in \mathbb{R}^n``. We obtain a feature matrix ``X\in\mathbb{R}^{n\times |A|}``.
- We use a Generalized Linear Model (GLM) for the encoder:
```math
\theta_a = \varphi_w(x_a) = w^T x_a + b \text{ with } w\in \mathbb{R}^n \text{ and } b\in\mathbb{R} \text{ learnable parameters}
```
- We solve the *easy problem* using the predicted ``\theta``:
```math
\begin{aligned}
y = \arg\max_y & \sum_{a\in A} \theta_a y_a &\\
s.t. & \sum_{a\in \delta^-(v)} y_a = \sum_{a\in \delta^+(v)} y_a, & \forall v \in V\backslash \{o, d\}\\
& \sum_{a\in \delta^-(v)} y_a = 1, & \forall v \in V\backslash \{o, d\}\\
& y_a \in \{0, 1\}, &\forall a\in A
\end{aligned}
```
=#

## Imports useful packages
using StochasticVehicleScheduling
using InferOpt
using Random, Test
using Flux
using UnicodePlots
using Statistics: mean
Random.seed!(1);

# ## Dataset creation

#=
We create a dataset with 50 training and 50 test instances.
Each instance has 20 tasks and 10 scenarios.
=#
nb_samples = 100
split_ratio = 0.5
nb_tasks = 20
nb_scenarios = 10

function train_test_split(X::AbstractVector, Y::AbstractVector, split_ratio::Real)
    @assert length(X) == length(Y) "X and Y have different lengths"
    nb_training_samples = length(X) - trunc(Int, length(X) * split_ratio)
    X_train, Y_train = X[1:nb_training_samples], Y[1:nb_training_samples]
    X_test, Y_test = X[(nb_training_samples + 1):end], Y[(nb_training_samples + 1):end]
    return (X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
end

X, Y = generate_dataset(
    nb_samples; nb_tasks=nb_tasks, nb_scenarios=nb_scenarios, nb_it=10_000
)
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, split_ratio)
nb_features = size(X_train[1].features, 1)
data_train, data_test = zip(X_train, Y_train), zip(X_test, Y_test);

# ## Training

# ### Initialization

# Initialize the GLM predictor:
model = Chain(Dense(nb_features => 1), vec)

#

## Define the full pipeline
pipeline(x) = easy_problem(model(x.features); instance=x);

#=
We can first compute the initial predictions of the model,
evaluate the corresponding average objective to the labeled solutions
=#
initial_pred = [pipeline(x) for x in X_test]
initial_mean_obj = mean(evaluate_solution(y, x) for (x, y) in zip(X_test, initial_pred))
ground_truth_obj = [evaluate_solution(y, x) for (x, y) in data_test]
ground_truth_mean_obj = mean(ground_truth_obj)
@info "Ground truth" ground_truth_mean_obj
@info "Initial" initial_mean_obj
initial_obj_gap = initial_mean_obj - ground_truth_mean_obj
@info "Difference" initial_obj_gap

#=
Let's see if we can reduce this gap by training the pipeline.
We choose a perturbed Fenchel-Young loss with parameters ``ε = 0.1`` and ``M = 5``
=#

## Loss function
ε = 0.1
M = 5
function maximizer(θ::AbstractVector; instance::Instance)
    return easy_problem(θ; instance=instance, model_builder=cbc_model)
end
loss = FenchelYoungLoss(PerturbedNormal(maximizer; ε=ε, M=M))
flux_loss(x, y) = loss(model(x.features), y.value; instance=x)
## Optimizer
opt = ADAM();

# ### Training loop

# We train our model for 200 epochs
nb_epochs = 50
function hamming_distance(x::AbstractVector, y::AbstractVector)
    return sum(x[i] != y[i] for i in eachindex(x))
end
training_losses, test_losses = Float64[], Float64[]
Δ_objective_history, hamming_distances = Float64[], Float64[]
for _ in 1:nb_epochs
    l = mean(flux_loss(x, y) for (x, y) in data_train)
    l_test = mean(flux_loss(x, y) for (x, y) in data_test)
    Y_pred = [maximizer(model(x.features); instance=x) for x in X_test]
    values = [evaluate_solution(y, x) for (x, y) in zip(X_test, Y_pred)]
    V = mean((v_pred - v) / v for (v_pred, v) in zip(values, ground_truth_obj))
    H = mean(hamming_distance(y_pred, y.value) for (y_pred, y) in zip(Y_pred, Y_test))
    push!(training_losses, l)
    push!(test_losses, l_test)
    push!(Δ_objective_history, V)
    push!(hamming_distances, H)

    Flux.train!(flux_loss, Flux.params(model), data_train, opt)
end

# ### Results

# #### Train and test losses
println(lineplot(training_losses; title="Training loss"))
println(lineplot(test_losses; title="Test loss"))
@info "Initial/final training losses" training_losses[1] training_losses[end]
@info "Initial/final test loss" test_losses[1] test_losses[end]

# #### Other test metrics
# Let's check the average objective gap
println(lineplot(Δ_objective_history; title="Objective difference"))
@info "Initial objective difference" Δ_objective_history[1]
@info "Final objective difference" Δ_objective_history[end]

# The average hamming distance also decreases
println(lineplot(hamming_distances; title="Hamming distance"))
@info "Initial/final average hamming distance" hamming_distances[1] hamming_distances[end]

# The training was a success.

# Some tests for CI
@test training_losses[end] < training_losses[1] / 3
#
@test test_losses[end] < test_losses[1] / 3
#
@test Δ_objective_history[end] < Δ_objective_history[1]
#
@test hamming_distances[end] < hamming_distances[1]
