using InferOptModelZoo.VSP
using InferOpt
using InferOpt.Testing

using Flux
using ProgressMeter
using Random

Random.seed!(67);

# ## Dataset creation

nb_samples = 100
nb_tasks = 20
nb_scenarios = 10
X, Y = VSP.generate_dataset(nb_samples; nb_tasks=nb_tasks, nb_scenarios=nb_scenarios, nb_it=10_000)

X_train, X_test = train_test_split(X)
Y_train, Y_test = train_test_split(Y)

data_train = InferOptDataset(; X=X_train, Y=Y_train)
data_test = InferOptDataset(; X=X_test, Y=Y_test)

# ## Training

# ### Initialization

# Initialize the GLM predictor:
nb_features = size(X_train[1].features, 1)
encoder = Chain(Dense(nb_features => 1), vec)
loss = FenchelYoungLoss(Perturbed(easy_problem; ε=0.1, M=5))

## Define the full pipeline
pipeline(x) = easy_problem(encoder(x.features), instance=x);
flux_loss(x, y) = loss(encoder(x.features), y.value; instance=x)

cost(y; instance) = evaluate_solution(y, instance)

## Optimization
opt = ADAM()

metrics = Dict(
    "loss" => Loss,
    "cost gap" => CostGap,
)
train_metrics = [metric("Train $name") for (name, metric) in metrics]
test_metrics = [metric("Test $name") for (name, metric) in metrics]

additional_info = (; cost)
trainer = InferOptTrainer(
    encoder,
    train_metrics, test_metrics,
    opt,
    flux_loss, pipeline,
    additional_info
)

function train_loop!(trainer, data_train, data_test, nb_epochs)
    @showprogress for _ in 1:nb_epochs
        compute_metrics!(trainer, data_train, data_test)
        (;X, Y) = data_train
        Flux.train!(trainer.flux_loss, Flux.params(trainer.encoder), zip(X, Y), trainer.opt)
    end
end

nb_epochs = 100
train_loop!(trainer, data_train, data_test, nb_epochs)

## Evaluation
plot_perf(trainer)
test_perf(trainer)
