# # Tutorial

#=
In this tutorial, we will use hybrid methods between Machine Learning (ML) and
Combinatorial Optimization (CO) from the package [`InferOpt.jl`](https://github.com/axelparmentier/InferOpt.jl).
The goal is to learn to approximate a hard problem (Stochastic Vehicle Scheduling) by an easier one (flow problem).
For more details, you can look at `InferOpt` [documentation](https://axelparmentier.github.io/InferOpt.jl/),
or its corresponding [paper](https://arxiv.org/pdf/2207.13513.pdf).
=#

## Imports useful packages and fix seed
using Flux
using Plots
using InferOpt
using Random
using Statistics: mean
using StochasticVehicleScheduling
using UnicodePlots
Random.seed!(1);

# ## Introduction

#=
Let `instance` be an instance of the Stochastic Vehicle Scheduling problem, and ``D = (V, A)``
its associated graph. For each arc ``a\in A``, we compute a features vector ``x_a\in \mathbb{R}^n``.
We obtain a feature matrix ``X\in\mathbb{R}^{n\times |A|}``. Let ``w\in\mathbb{R}^n`` a weight vector.
=#

nb_tasks = 10
nb_scenarios = 10
instance = create_random_instance(; nb_tasks, nb_scenarios)
@info "Example instance" instance.graph instance.features

#=
We define the following Generalized Linear Model (GLM) ``\varphi_w``, that let us compute arcs
weights ``\theta_a\in \mathcal R`` for each arc $a\in A$.
```math
\theta_a = \varphi_w(x_a) = w^\top x_a
```
=#

nb_features = size(instance.features, 1)
φ_w = Chain(Dense(nb_features => 1; bias=false), vec)  # Note: we use methods from Flux to build the GLM, w is set randomly

# We can use the GLM to compute arcs weights θ
θ = φ_w(instance.features)
θ'

#=
Then, we define what we call the *easy problem* as the following linear program:
```math
\begin{aligned}
y = \arg\max_y & \sum_{a\in A} \theta_a y_a &\\
s.t. & \sum_{a\in \delta^-(v)} y_a = \sum_{a\in \delta^+(v)} y_a, & \forall v \in V\backslash \{o, d\}\\
& \sum_{a\in \delta^-(v)} y_a = 1, & \forall v \in V\backslash \{o, d\}\\
& y_a \in \{0, 1\}, &\forall a\in A
\end{aligned}
```
Integrity contraints can be dropped because we have a flow polytope. Therefore, the *easy problem*
can be solved easily, which is done by the [`easy_problem`](@ref) method
(note: we need to give `instance` as input because we need to know the graph in order to build the constraints).
=#

y = easy_problem(θ; instance=instance)
y'

#=
By applying it to arcs weights ``\theta``, we obtain the optimal solution ``y`` of the *easy problem*,
which is also a feasible solution of the initial stochastic vehicle scheduling `instance` !
Let's evaluate it:
=#

evaluate_solution(y, instance)

#=
The objective value does not seem very good. Let's compare it to the value of the
optimal solution of `instance`:
=#

_, y_opt = solve_scenarios(instance)
evaluate_solution(y_opt, instance)

#=
This is not good...
Can we do better? Is it possible to find ``w`` that predicts ``\theta`` that gives good solutions ?

The answer is yes, we can use tools from `InferOpt` in order to learn parameters ``w`` that yield good solutions
for any instance.

First of all, let's assemble together our full pipeline:
```math
\xrightarrow[\text{Instance}]{X}
\fbox{Encoder $\varphi_w$}
\xrightarrow[\text{Cost vector}]{\theta \in \mathbb{R}^{d(x)}}
\fbox{Easy problem}
\xrightarrow[\text{Solution}]{y \in \mathcal{Y}(x)}
```
=#

pipeline(x::Instance) = easy_problem(φ_w(x.features); instance=x);

# ## Dataset creation

#=
In order to learn something, we first need to create a dataset containing instances and
corresponding solutions. We create a training dataset, and a validation dataset to evaluate teh results.
=#

nb_train_samples = 25
nb_val_samples = 10

X_train = [create_random_instance(; nb_tasks, nb_scenarios) for _ in 1:nb_train_samples]
Y_train = [solve_scenarios(x)[2] for x in X_train]

X_val = [create_random_instance(; nb_tasks, nb_scenarios) for _ in 1:nb_val_samples]
Y_val = [solve_scenarios(x)[2] for x in X_val]

data_train, data_val = zip(X_train, Y_train), zip(X_val, Y_val);

#=
We can evaluate our current pipeline by computing the average gap with the optimal solutions:
=#
initial_pred = [pipeline(x) for x in X_val]
initial_obj = [evaluate_solution(y, x) for (x, y) in zip(X_val, initial_pred)]
ground_truth_obj = [evaluate_solution(y, x) for (x, y) in data_val]

initial_average_gap = mean((initial_obj .- ground_truth_obj) ./ ground_truth_obj .* 100)
@info "Initial gap = $initial_average_gap%"

# The gap very high, let's see if we can find a good ``w`` to reduce it.

# ## Learning by imitation

# ### Regularization
#=
In order to train our model by using gradient descent, we need to be able to differentiate
through `easy_problem`. The problem is that, as a combinatorial allgorithm, it's pieciwise
constant, therefore gradient descent won't work at all:
gradient(θ -> easy_problem(θ; instance=instance), θ)

That's why we need to regularize the easy problem. Here we choose the `PerturbedAdditive`
regularization from `InferOpt`.
=#

regularized_predictor = PerturbedAdditive(easy_problem; ε=0.1, nb_samples=10)

#=
Instead of returning a binary solution, this wrapper around `easy_problem` takes the same inputs
but returns a smooth a continuous solution.
=#

y_pred = regularized_predictor(rand(length(θ)); instance=instance)
y_pred[y_pred .>= 0.1 .&& y_pred .<= 0.9]'

# ### Loss function

loss = FenchelYoungLoss(regularized_predictor)
flux_loss(x, y) = loss(φ_w(x.features), y.value; instance=x)

# ### Training loop
## Optimizer
opt = Adam();

# We train our model for 25 epochs
nb_epochs = 25
training_losses, val_losses = Float64[], Float64[]
objective_gap_history = Float64[]
for _ in 1:nb_epochs
    l = mean(flux_loss(x, y) for (x, y) in data_train)
    l_test = mean(flux_loss(x, y) for (x, y) in data_val)
    Y_pred = [pipeline(x) for x in X_val]
    values = [evaluate_solution(y, x) for (x, y) in zip(X_val, Y_pred)]
    V = mean((v_pred - v) / v * 100 for (v_pred, v) in zip(values, ground_truth_obj))
    push!(training_losses, l)
    push!(val_losses, l_test)
    push!(objective_gap_history, V)

    Flux.train!(flux_loss, Flux.params(φ_w), data_train, opt)
end

# ### Results

# #### Train and test losses
plot(training_losses; label="Training loss")
plot!(val_losses; label="Validation loss")

# #### Other test metrics
# Let's check the average objective gap
plot(objective_gap_history; title="Objective gap")

#

@info "Initial objective gap = $(objective_gap_history[1])%"
@info "Final objective gap = $(objective_gap_history[end])%"

# The training was a success.
