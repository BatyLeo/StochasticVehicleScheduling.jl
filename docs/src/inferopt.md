```@meta
EditURL = "<unknown>/docs/src/literate/inferopt.jl"
```

# InferOpt tutorial

In this tutorial, we will use hybrid methods between Machine Learning (ML) and
Combinatorial Optimization (CO) from the package [`InferOpt.jl`](https://github.com/axelparmentier/InferOpt.jl).
The goal is to learn to approximate a hard problem (Stochastic Vehicle Scheduling Problem) by an easier one (Vehicle Scheduling Problem).
For more details, you can look at the `InferOpt.jl` [documentation](https://axelparmentier.github.io/InferOpt.jl/),
or its corresponding [paper](https://arxiv.org/pdf/2207.13513.pdf).

````@example inferopt
# Imports useful packages and fix seed
using Flux
using Plots
using InferOpt
using Random
using Statistics: mean
using StochasticVehicleScheduling
Random.seed!(1);
nothing #hide
````

## Introduction

Let `instance` be an instance of the Stochastic Vehicle Scheduling problem, and ``D = (V, A)``
its associated graph. For each arc ``a\in A``, we compute a features vector ``x_a\in \mathbb{R}^n``.
We obtain a feature matrix ``X\in\mathbb{R}^{n\times |A|}``. For more details about the features computation, see [Features](@ref).

````@example inferopt
nb_tasks = 10
nb_scenarios = 10
instance = create_random_instance(; nb_tasks, nb_scenarios)
@info "Example instance" instance.graph instance.features
````

Let ``w\in\mathbb{R}^n`` a weight vector. We define the following Generalized Linear Model (GLM) ``\varphi_w``, that let us compute arcs
weights ``\theta_a\in \mathcal R`` for each arc $a\in A$.
```math
\theta_a = \varphi_w(x_a) = w^\top x_a
```

````@example inferopt
nb_features = size(instance.features, 1)
φ_w = Chain(Dense(nb_features => 1; bias=false), vec)  # Note: we use methods from Flux to build the GLM, w is initialized randomly by default
````

We can use the GLM to compute arcs weights θ

````@example inferopt
θ = φ_w(instance.features)
θ'
````

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

By applying it to arcs weights ``\theta``, we obtain the optimal solution ``y`` of the *easy problem*,
which is also a feasible solution of the initial stochastic vehicle scheduling `instance`:

````@example inferopt
y = easy_problem(θ; instance=instance)
y'
````

Let's evaluate it:

````@example inferopt
evaluate_solution(y, instance)
````

The objective value does not seem very good. Let's compare it to the value of the
optimal solution of `instance`, using the exact algorrithm described in [MIP formulation](@ref):

````@example inferopt
_, y_opt = solve_scenarios(instance)
evaluate_solution(y_opt, instance)
````

This is not good, our prediction is far off the optimal solution 😟

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

````@example inferopt
pipeline(x) = easy_problem(φ_w(x.features); instance=x);
nothing #hide
````

## Dataset creation

In order to learn something, we first need to create a dataset containing instances and
corresponding solutions. We create a training dataset, and a validation dataset to evaluate the results.

````@example inferopt
nb_train_samples = 25
nb_val_samples = 25

X_train = [create_random_instance(; nb_tasks, nb_scenarios) for _ in 1:nb_train_samples]
Y_train = [solve_scenarios(x)[2] for x in X_train]

X_val = [create_random_instance(; nb_tasks, nb_scenarios) for _ in 1:nb_val_samples]
Y_val = [solve_scenarios(x)[2] for x in X_val]

data_train, data_val = zip(X_train, Y_train), zip(X_val, Y_val);
nothing #hide
````

We can evaluate our current pipeline by computing the average gap with the optimal solutions:

````@example inferopt
initial_pred = [pipeline(x) for x in X_val]
initial_obj = [evaluate_solution(y, x) for (x, y) in zip(X_val, initial_pred)]
ground_truth_obj = [evaluate_solution(y, x) for (x, y) in data_val]

initial_average_gap = mean((initial_obj .- ground_truth_obj) ./ ground_truth_obj .* 100)
@info "Initial gap ≃ $(round(initial_average_gap; digits=2))%"
````

The gap very high, let's see if we can learn a good ``w`` that reduces it.

## Learning by imitation

### Regularization
In order to train our model by using gradient descent, we need to be able to differentiate
through `easy_problem` with automatic differenciation tools. The problem is that, as a combinatorial algorithm, it's pieciwise
constant, therefore gradient descent won't work at all:

````@example inferopt
try
    gradient(θ -> easy_problem(θ; instance=instance), θ)
catch e
    @error e
end
````

That's why we need to regularize the easy problem. Here we choose the `PerturbedAdditive`
regularization from `InferOpt.jl`.

````@example inferopt
regularized_predictor = PerturbedAdditive(easy_problem; ε=0.1, nb_samples=10)
````

Instead of returning a binary solution, this wrapper around `easy_problem` takes the same inputs
but returns a smooth a continuous solution.

````@example inferopt
y_pred = regularized_predictor(rand(length(θ)); instance=instance)
y_pred[y_pred .>= 0.1 .&& y_pred .<= 0.9]'
````

### Loss function

We can now choose a differentiable loss function from InferOpt's toolbox.
We choose a `FenchelYoungLoss`, that wraps our perturbed maximizer.

````@example inferopt
loss = FenchelYoungLoss(regularized_predictor)
````

We can evaluate our prediction ``\theta`` respect to the optimal solution `y_opt`

````@example inferopt
loss(θ, y_opt.value; instance=instance)
````

And compute its gradient respect to ``\theta``

````@example inferopt
gradient(θ -> loss(θ, y_opt.value; instance=instance), θ)
````

We now have all the tools in order to learn our pipeline using gradient descent and automatic differentiation !

### Training loop

We train our model using the `Flux.jl` library, and tha Adam optimizer.

````@example inferopt
flux_loss(x, y) = loss(φ_w(x.features), y.value; instance=x)
# Optimizer
opt = Adam();
nothing #hide
````

We train our model for 25 epochs

````@example inferopt
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
````

### Results

#### Train and test losses
Let's check our pipeline performance, by plotting training and validation losses.

````@example inferopt
plot(training_losses; label="Training loss")
plot!(val_losses; label="Validation loss")
````

The loss decreased, which is a good sign.

#### Other test metrics
We can also check the average optimality gap as another metric.

````@example inferopt
plot(objective_gap_history; title="Objective gap")
````

The optimality gap also decreased:

````@example inferopt
@info "Initial objective gap ≃ $(round(objective_gap_history[1], digits=2))%"
@info "Final objective gap ≃ $(round(objective_gap_history[end], digits=2))%"
````

Its value is now less than 1%. The training was a success 🎉

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

