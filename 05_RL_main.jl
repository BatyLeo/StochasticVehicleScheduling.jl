include("00_config.jl")
include("04_RL_algorithms.jl")

using Flux
using Flux.Optimise
using Gurobi: Gurobi
using Distributions
using LinearAlgebra
using InferOpt: PerturbedAdditive, FenchelYoungLoss, Pushforward
using ProgressMeter: @showprogress
using UnicodePlots: lineplot

# Load data
data = JLD2.load(dataset_path);
dataset_25 = data["dataset_25"];
dataset_50 = data["dataset_50"];
dataset_100 = data["dataset_100"];

train_set_25, val_set_25, test_set_25 = splitobs(dataset_25; at=(50, 50));
train_set_50, val_set_50, test_set_50 = splitobs(dataset_50; at=(50, 50));
train_set_100, val_set_100, test_set_100 = splitobs(dataset_100; at=(50, 50));

b = StochasticVehicleSchedulingBenchmark()
model = generate_statistical_model(b; seed=0)
maximizer = generate_maximizer(b)

# RL training
IL_model, train_IL, val_IL, gaps_IL, losses_IL = IL_training(deepcopy(model), train_set_25, val_set_25;
    nb_epochs=200, batch_size=4, no_samples=20, sigma_values=[0.5, 0.05], lr_values = [5e-3, 1e-3], soft=true, temp_values=[10.0, 0.1]
)
il_runtime = @timed IL_training(deepcopy(model), train_set_25, val_set_25; nb_epochs=200, batch_size=4, no_samples=20, sigma_values=[0.5, 0.05], lr_values = [5e-3, 1e-3], soft=true, temp_values=[10.0, 0.1])
il_runtime.time
# temp: 1e0, 1e2, 1e3, 1e4, 1e5
# 1st best: sigma = [0.1, 0.01], lr = [0.01, 0.005], temp = [10000.0, 100.0]
# 2nd best: sigma = [0.5, 0.05], lr = [1e-2, 5e-3], temp = 1e5

function IL_test(sigma_steps, lr_steps, temp_steps, seeds; soft=true)
    final_train = []
    final_val = []
    all_rews = []
    for i in sigma_steps
        for j in lr_steps
            for k in temp_steps
                for s in seeds
                    model = generate_statistical_model(b; seed=s)
                    IL_model, train_IL, val_IL, gaps_IL, losses_IL = IL_training(deepcopy(model), train_set_25, val_set_25;
                        nb_epochs=200, batch_size=4, no_samples=20, sigma_values=i, lr_values = j, soft=true, temp_values=k
                    )
                    final_tr = [evaluate_solution(maximizer(IL_model(i.x); instance=i.instance), i.instance) for i in train_set_25]
                    final_te = [evaluate_solution(maximizer(IL_model(i.x); instance=i.instance), i.instance) for i in test_set_25]
                    push!(all_rews, (sigma=i, lr=j, temp=k, seed=s, model=deepcopy(IL_model), train_rew=train_IL, val_rew=val_IL, train_final=final_tr, test_final=final_te))
                    push!(final_train, train_IL[end])
                    push!(final_val, val_IL[end])
                end
            end
        end
    end
    return final_train, final_val, all_rews
end

train_rews, val_rews, all_rews = IL_test(
    [[1.0, 0.5], [0.5, 0.05], [0.1, 0.01]],
    [[1e-2, 5e-3], [5e-3, 1e-3], [1e-3, 5e-4]],
    [[1e5, 1e0], [1e4, 1e2], [1e5, 1e3]],
    [0]
)
il_train, il_val, il_results = IL_test([[0.1, 0.01]], [[0.01, 0.005]], [[10000.0, 100.0]], [1, 2, 3, 4, 5, 6, 7, 8, 9])
JLD2.jldsave("logs/svsp_il_random_seeds.jld2"; results=il_results)

train_idx = partialsortperm(train_rews, rev=false, 1:4)
val_idx = partialsortperm(val_rews, rev=false, 1:4)
mean_rews = [(train_rews[i] + val_rews[i]) / 2 for i in 1:length(train_rews)]
mean_idx = partialsortperm(mean_rews, rev=false, 1:4)
train_rews[train_idx]
val_rews[val_idx]
all_rews[20]
IL_model = all_rews[20].model
train_IL = all_rews[20].train_rew
val_IL = all_rews[20].val_rew

PPO_model, train_PPO, val_PPO, gaps_PPO, losses_PPO = PPO_training(deepcopy(model), train_set_25, val_set_25;
    nb_epochs=200, batch_size=4, clip=0.2, sigma_values=[0.5, 0.5], lr_values = [1e-2, 1e-2]
)
ppo_runtime = @timed PPO_training(deepcopy(model), train_set_25, val_set_25; nb_epochs=200, batch_size=4, clip=0.2, sigma_values=[0.5, 0.5], lr_values = [1e-2, 1e-2])
ppo_runtime.time
# nb_epochs=400, batch_size=4, clip=0.2, sigma_values=[0.5, 0.1], lr_values = [1e-2, 1e-2]

function PPO_runs(sigma_steps, lr_steps, seeds; soft=true)
    final_train = []
    final_val = []
    all_rews = []
    for i in sigma_steps
        for j in lr_steps
            for s in seeds
                model = generate_statistical_model(b; seed=s)
                PPO_model, train_PPO, val_PPO, gaps_PPO, losses_PPO = PPO_training(deepcopy(model), train_set_25, val_set_25;
                    nb_epochs=200, batch_size=4, clip=0.2, sigma_values=i, lr_values = j
                )
                final_tr = [evaluate_solution(maximizer(PPO_model(i.x); instance=i.instance), i.instance) for i in train_set_25]
                final_te = [evaluate_solution(maximizer(PPO_model(i.x); instance=i.instance), i.instance) for i in test_set_25]
                push!(all_rews, (sigma=i, lr=j, seed=s, model=deepcopy(PPO_model), train_rew=train_PPO, val_rew=val_PPO, train_final=final_tr, test_final=final_te))
                push!(final_train, train_PPO[end])
                push!(final_val, val_PPO[end])
            end
        end
    end
    return final_train, final_val, all_rews
end

ppo_train, ppo_val, ppo_results = PPO_runs([[0.5, 0.1]], [[1e-2, 1e-2]], [1, 2, 3, 4, 5, 6, 7, 8, 9])
JLD2.jldsave("logs/svsp_ppo_random_seeds.jld2"; results=ppo_results)

# RL tests
compute_gap(b, train_set_25, IL_model, maximizer) # 0.0028470525704966253
compute_gap(b, test_set_100, IL_model, maximizer) # 0.004305327212030902
IL_train = mean([evaluate_solution(maximizer(IL_model(i.x); instance=i.instance), i.instance) for i in train_set_25]) # 6903.598711202943
IL_testrew = mean([evaluate_solution(maximizer(IL_model(i.x); instance=i.instance), i.instance) for i in test_set_100]) # 20818.001812770195
compute_gap(b, train_set_25, PPO_model, maximizer) # 0.0050
compute_gap(b, test_set_100, PPO_model, maximizer) # 0.0294
PPO_train = mean([evaluate_solution(maximizer(PPO_model(i.x); instance=i.instance), i.instance) for i in train_set_25]) # 6919.344525402182
PPO_test = mean([evaluate_solution(maximizer(PPO_model(i.x); instance=i.instance), i.instance) for i in test_set_100]) # 21342.011725650435

IL_final_train_rew = [evaluate_solution(maximizer(IL_model(i.x); instance=i.instance), i.instance) for i in train_set_25]
IL_final_test_rew = [evaluate_solution(maximizer(IL_model(i.x); instance=i.instance), i.instance) for i in test_set_25]
JLD2.jldsave("logs/svsp_il_best_model.jld2"; model=IL_model, train_rew=train_IL, val_rew=val_IL, train_final=IL_final_train_rew, test_final=IL_final_test_rew)

PPO_final_train_rew = [evaluate_solution(maximizer(PPO_model(i.x); instance=i.instance), i.instance) for i in train_set_25]
PPO_final_test_rew = [evaluate_solution(maximizer(PPO_model(i.x); instance=i.instance), i.instance) for i in test_set_25]
JLD2.jldsave("logs/svsp_ppo_best_model.jld2"; model=PPO_model, train_rew=train_PPO, val_rew=val_PPO, train_final=PPO_final_train_rew, test_final=PPO_final_test_rew)

data = JLD2.load("logs/svsp_sl_random_seeds.jld2")["results"]
all_results = []
for d in data
    model = d.model
    final_tr = [evaluate_solution(maximizer(model(i.x); instance=i.instance), i.instance) for i in train_set_25]
    final_te = [evaluate_solution(maximizer(model(i.x); instance=i.instance), i.instance) for i in test_set_25]
    push!(all_results, (seed=d.seed, model=d.model, train_rew=d.train_rew, val_rew=d.val_rew, train_final=final_tr, test_final=final_te))
    # push!(all_results, (sigma=d.seed, lr=d.lr, seed=d.seed, model=d.model, train_rew=d.train_rew, val_rew=d.val_rew, train_final=final_tr, test_final=final_te))
    # push!(all_results, (sigma=d.seed, lr=d.lr, temp=d.temp, seed=d.seed, model=d.model, train_rew=d.train_rew, val_rew=d.val_rew, train_final=final_tr, test_final=final_te))
end
JLD2.jldsave("logs/svsp_sl_random_seeds.jld2"; results=all_results)
# (seed=s, model=sl_model, train_rew=train_hist, val_rew=val_hist, train_final=final_tr, test_final=final_te)