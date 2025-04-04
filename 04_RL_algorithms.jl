include("00_config.jl")

using Flux
using Flux.Optimise
using Gurobi: Gurobi
using Distributions
using LinearAlgebra
using InferOpt: PerturbedAdditive, FenchelYoungLoss, Pushforward
using ProgressMeter: @showprogress
using UnicodePlots: lineplot

function reward_comparison(train_rew, val_rew)
    means = [(train_rew[i] + val_rew[i]) / 2 for i in 1:length(train_rew)]
    last_mean = means[end]  # Mean of the last two elements
    return last_mean == minimum(means)  # Check if it's the smallest mean
end

function IL_training(
    model, train_set, val_set; nb_epochs=100, batch_size = 10, no_samples = 20, sigma_values=[0.05, 0.05], lr_values = [1e-3, 1e-3], soft=false, temp=1.0
)
    loss = FenchelYoungLoss(PerturbedAdditive(maximizer; ε=1.0, nb_samples=20))
    opt = Flux.Optimise.Adam(lr_values[1])
    lr_step = (lr_values[1] - lr_values[2]) / nb_epochs
    train_costs = Float64[]
    val_costs = Float64[]
    gap_history = Float64[]
    best_model = deepcopy(model)
    best_episode = 0
    opt_state = Flux.setup(opt, model)

    prob(θ, eps) = MvNormal(θ, eps * I)
    sigma = sigma_values[1]
    sigma_step = (sigma_values[1] - sigma_values[2]) / nb_epochs

    losses = Float64[]
    for e in 1:nb_epochs
        push!(train_costs, mean([evaluate_solution(maximizer(model(i.x); instance=i.instance), i.instance) for i in train_set]),)
        push!(val_costs, mean([evaluate_solution(maximizer(model(i.x); instance=i.instance), i.instance) for i in val_set]),)
        push!(gap_history, compute_gap(b, val_set, model, maximizer),)
        @info e, "sigma:", sigma, "lr", lr_values[1], "train:", train_costs[end], "val:", val_costs[end], "gap:", gap_history[end]
        if reward_comparison(train_costs, val_costs)
            best_model = deepcopy(model)
            best_episode = e
        end

        batches = Flux.DataLoader(train_set; batchsize=batch_size, shuffle=true)

        for batch in batches
            best_solutions = []
            for b in batch
                θ = model(b.x)
                η = rand(prob(θ, sigma), no_samples)
                solutions = [maximizer(θ; instance=b.instance)]
                values = [evaluate_solution(solutions[end], b.instance)]
                for i in 1:no_samples
                    push!(solutions, maximizer(η[:, i]; instance=b.instance),)
                    push!(values, evaluate_solution(solutions[end], b.instance),)
                end
                if soft
                    values = values ./ (-temp)
                    lse = logsumexp(values)
                    probs = exp.(values .- lse)
                    best_action = sum(probs .* solutions)
                    # any(isnan.(best_action)) ? best_action = solutions[argmax(values)] : nothing
                else
                    best_action = solutions[argmin(values)]
                end
                push!(best_solutions, best_action,)
            end

            val, grads = Flux.withgradient(model) do m
                sum([loss(m(b.x), b.y_true; instance=b.instance) for b in batch])
            end
            Flux.update!(opt_state, model, grads[1])
            push!(losses, val,)
        end
        sigma = max(sigma - sigma_step, sigma_values[2])
        lr = opt.eta
        opt.eta = max(lr - lr_step, lr_values[2])
    end

    push!(train_costs, mean([evaluate_solution(maximizer(best_model(i.x); instance=i.instance), i.instance) for i in train_set]),)
    push!(val_costs, mean([evaluate_solution(maximizer(best_model(i.x); instance=i.instance), i.instance) for i in val_set]),)
    push!(gap_history, compute_gap(b, val_set, model, maximizer),)
    @info "final train:", train_costs[end], "final val:", val_costs[end], "final gap:", gap_history[end], "best_episode:", best_episode
    return best_model, train_costs, val_costs, gap_history, losses
end

function PPO_training(
    model, train_set, val_set; nb_epochs=100, batch_size = 10, clip = 0.2, sigma_values=[0.05, 0.05], lr_values = [1e-3, 1e-3]
)
    opt = Flux.Optimise.Adam(lr_values[1])
    lr_step = (lr_values[1] - lr_values[2]) / nb_epochs
    train_costs = Float64[]
    val_costs = Float64[]
    gap_history = Float64[]
    best_model = deepcopy(model)
    best_episode = 0
    opt_state = Flux.setup(opt, model)

    prob(θ, eps) = MvNormal(θ, eps * I)
    sigma = sigma_values[1]
    sigma_step = (sigma_values[1] - sigma_values[2]) / nb_epochs

    losses = Float64[]
    for e in 1:nb_epochs
        push!(train_costs, mean([evaluate_solution(maximizer(model(i.x); instance=i.instance), i.instance) for i in train_set]),)
        push!(val_costs, mean([evaluate_solution(maximizer(model(i.x); instance=i.instance), i.instance) for i in val_set]),)
        push!(gap_history, compute_gap(b, val_set, model, maximizer),)
        @info e, "sigma:", sigma, "lr", lr_values[1], "train:", train_costs[end], "val:", val_costs[end], "gap:", gap_history[end]
        if reward_comparison(train_costs, val_costs)
            best_model = deepcopy(model)
            best_episode = e
        end

        batches = Flux.DataLoader(train_set; batchsize=batch_size, shuffle=true)

        for batch in batches
            thetas = [model(b.x) for b in batch]
            etas = [rand(prob(thetas[j], sigma)) for j in 1:length(batch)]
            advantages = [evaluate_solution(maximizer(thetas[j]; instance=batch[j].instance), batch[j].instance) - evaluate_solution(maximizer(etas[j]; instance=batch[j].instance), batch[j].instance) for j in 1:length(batch)]
            
            val, grads = Flux.withgradient(model) do m
                old_probs = [pdf(prob(thetas[j], sigma), etas[j]) for j in 1:length(batch)]
                new_probs = [pdf(prob(m(batch[j].x), sigma), etas[j]) for j in 1:length(batch)]
                ratio_unclipped = [new_probs[j] / old_probs[j] for j in 1:length(batch)]
                ratio_clipped = clamp.(ratio_unclipped, 1-clip, 1+clip)
                return -mean(min.(ratio_unclipped .* advantages, ratio_clipped .* advantages))
            end
            Flux.update!(opt_state, model, grads[1])
            push!(losses, val,)
        end
        sigma = max(sigma - sigma_step, sigma_values[2])
        lr = opt.eta
        opt.eta = max(lr - lr_step, lr_values[2])
    end

    push!(train_costs, mean([evaluate_solution(maximizer(best_model(i.x); instance=i.instance), i.instance) for i in train_set]),)
    push!(val_costs, mean([evaluate_solution(maximizer(best_model(i.x); instance=i.instance), i.instance) for i in val_set]),)
    push!(gap_history, compute_gap(b, val_set, model, maximizer),)
    @info "final train:", train_costs[end], "final val:", val_costs[end], "final gap:", gap_history[end], "best_episode:", best_episode
    return best_model, train_costs, val_costs, gap_history, losses
end