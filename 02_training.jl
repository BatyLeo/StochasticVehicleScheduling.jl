include("00_config.jl")

using Flux: Flux, Adam, Descent
using Gurobi: Gurobi
using InferOpt: PerturbedAdditive, FenchelYoungLoss, Pushforward
using ProgressMeter: @showprogress
using UnicodePlots: lineplot

data = JLD2.load(dataset_path);
dataset_25 = data["dataset_25"];
dataset_50 = data["dataset_50"];
dataset_100 = data["dataset_100"];

b = StochasticVehicleSchedulingBenchmark()

# Statistical model
model = generate_statistical_model(b; seed=0)

# CO layer and losses
maximizer = generate_maximizer(b)

perturbed = PerturbedAdditive(maximizer; ε=1.0, nb_samples=20)
fyl_loss = FenchelYoungLoss(perturbed)

cost(y; instance) = evaluate_solution(y, instance)
perturbed_regret = Pushforward(perturbed, cost)

supervised_loss(sample, m) = fyl_loss(m(sample.x), sample.y_true; instance=sample.instance)
experience_loss(sample, m) = perturbed_regret(m(sample.x); instance=sample.instance)

sample = dataset_25[1]
sample.x
sample.y_true
sample.instance
model(sample.x)
maximizer(model(sample.x); instance=sample.instance)
cost(sample.y_true; sample.instance)
cost(maximizer(model(sample.x); instance=sample.instance); sample.instance)
supervised_loss(sample, model)
experience_loss(sample, model)

function train_model!(
    model, maximizer, train_set, val_set, loss; nb_epochs=50, optimizer=Adam()
)
    train_costs = []
    val_costs = []
    loss_history = [mapreduce(d -> loss(d, model), +, train_set) / numobs(train_set)]
    gap_history = [compute_gap(b, val_set, model, maximizer)]
    best_model = deepcopy(model)
    best_gap = gap_history[1]

    opt_state = Flux.setup(optimizer, model)
    for e in 1:nb_epochs
        push!(train_costs, mean([evaluate_solution(maximizer(model(i.x); instance=i.instance), i.instance) for i in train_set]),)
        push!(val_costs, mean([evaluate_solution(maximizer(model(i.x); instance=i.instance), i.instance) for i in val_set]),)
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

    push!(train_costs, mean([evaluate_solution(maximizer(best_model(i.x); instance=i.instance), i.instance) for i in train_set]),)
    push!(val_costs, mean([evaluate_solution(maximizer(best_model(i.x); instance=i.instance), i.instance) for i in val_set]),)
    return best_model, train_costs, val_costs, loss_history, gap_history
end

supervised_model, supervised_train, supervised_val, supervised_loss_history, supervised_gap_history = train_model!(
    deepcopy(model), maximizer, train_set_25, val_set_25, supervised_loss; nb_epochs=200
)
experience_model, experience_train, experience_val, experience_loss_history, experience_gap_history = train_model!(
    deepcopy(model), maximizer, train_set_25, val_set_25, experience_loss; nb_epochs=100
)
sl_runtime = @timed train_model!(deepcopy(model), maximizer, train_set_25, val_set_25, supervised_loss; nb_epochs=200)
sl_runtime.time

function sl_runs(seeds)
    all_results = []
    for s in seeds
        model = generate_statistical_model(b; seed=s)
        sl_model, train_hist, val_hist, supervised_loss_history, supervised_gap_history = train_model!(
            deepcopy(model), maximizer, train_set_25, val_set_25, supervised_loss; nb_epochs=200
        )
        final_tr = [evaluate_solution(maximizer(sl_model(i.x); instance=i.instance), i.instance) for i in train_set_25]
        final_te = [evaluate_solution(maximizer(sl_model(i.x); instance=i.instance), i.instance) for i in test_set_25]
        push!(all_results, (seed=s, model=sl_model, train_rew=train_hist, val_rew=val_hist, train_final=final_tr, test_final=final_te))
    end
    return all_results
end

sl_results = sl_runs([1, 2, 3, 4, 5, 6, 7, 8, 9])
JLD2.jldsave("logs/svsp_sl_random_seeds.jld2"; results=sl_results)

JLD2.jldsave("logs/SL_trained.jld2"; model=supervised_model, gaps=supervised_gap_history)
JLD2.jldsave("logs/RM_trained.jld2"; model=experience_model, gaps=experience_gap_history)
# JLD2.jldsave(
#     results_path;
#     supervised_model,
#     experience_model,
#     supervised_loss_history,
#     experience_loss_history,
#     supervised_gap_history,
#     experience_gap_history,
# )
