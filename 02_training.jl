include("00_config.jl")

using Flux: Flux, Adam, Descent
using Gurobi: Gurobi
using InferOpt: PerturbedAdditive, FenchelYoungLoss, Pushforward
using ProgressMeter: @showprogress
using UnicodePlots: lineplot

data = JLD2.load(dataset_path);
train_set_25 = data["train_set_25"];
val_set_25 = data["val_set_25"];

b = StochasticVehicleSchedulingBenchmark()

# Statistical model
model = generate_statistical_model(b; seed=0)

# CO layer and losses
maximizer = generate_maximizer(b; model_builder)

perturbed = PerturbedAdditive(maximizer; ε=1.0, nb_samples=20)
fyl_loss = FenchelYoungLoss(perturbed)

cost(y; instance) = evaluate_solution(y, instance)
perturbed_regret = Pushforward(perturbed, cost)

supervised_loss(sample, m) = fyl_loss(m(sample.x), sample.y_true; instance=sample.instance)
experience_loss(sample, m) = perturbed_regret(m(sample.x); instance=sample.instance)

function train_model!(
    model, maximizer, train_set, val_set, loss; nb_epochs=50, optimizer=Adam()
)
    loss_history = [mapreduce(d -> loss(d, model), +, train_set) / numobs(train_set)]
    gap_history = [compute_gap(b, val_set, model, maximizer)]
    best_model = deepcopy(model)
    best_gap = gap_history[1]

    opt_state = Flux.setup(optimizer, model)
    for e in 1:nb_epochs
        loss_sum = 0.0
        g = round(gap_history[end] * 100; digits=1)
        @showprogress desc = "Epoch $e gap: $g% |" for sample in train_set
            val, grads = Flux.withgradient(model) do m
                loss(sample, m)
            end
            Flux.update!(opt_state, model, grads[1])
            loss_sum += val
        end
        push!(loss_history, loss_sum / numobs(train_set))
        push!(gap_history, compute_gap(b, val_set, model, maximizer))
        if gap_history[end] < best_gap
            best_model = deepcopy(model)
            best_gap = gap_history[end]
        end
    end

    return best_model, loss_history, gap_history
end

supervised_model, supervised_loss_history, supervised_gap_history = train_model!(
    deepcopy(model), maximizer, train_set_25, val_set_25, supervised_loss; nb_epochs=50
)
experience_model, experience_loss_history, experience_gap_history = train_model!(
    deepcopy(model), maximizer, train_set_25, val_set_25, experience_loss; nb_epochs=100
)

JLD2.jldsave(
    results_path;
    supervised_model,
    experience_model,
    supervised_loss_history,
    experience_loss_history,
    supervised_gap_history,
    experience_gap_history,
)
