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
    nb_epochs=200, batch_size=4, no_samples=20, sigma_values=[0.5, 0.05], lr_values = [5e-3, 1e-3], soft=true, temp=1e0
)
# temp: 1e0, 1e2, 1e3, 1e4, 1e5
# sigma = [0.5, 0.05], lr = [1e-2, 5e-3], temp = 1e5
# nb_epochs=200, batch_size=4, no_samples=20, sigma_values=[0.5, 0.05], lr_values = [1e-3, 1e-3]

function IL_test(sigma_steps, lr_steps, temp_steps; soft=true)
    final_train = []
    final_val = []
    all_rews = []
    for i in sigma_steps
        for j in lr_steps
            for k in temp_steps
                IL_model, train_IL, val_IL, gaps_IL, losses_IL = IL_training(deepcopy(model), train_set_25, val_set_25;
                    nb_epochs=200, batch_size=4, no_samples=20, sigma_values=i, lr_values = j, soft=true, temp=k
                )
                push!(all_rews, (sigma=i, lr=j, temp=k, model=deepcopy(IL_model), train_rew=train_IL, val_rew=val_IL))
                push!(final_train, train_IL[end])
                push!(final_val, val_IL[end])
            end
        end
    end
    return final_train, final_val, all_rews
end

train_rews, val_rews, all_rews = IL_test(
    [[1.0, 0.5], [0.5, 0.05], [0.1, 0.01]],
    [[1e-2, 5e-3], [5e-3, 1e-3], [1e-3, 5e-4]],
    [1e0, 1e2, 1e3, 1e4, 1e5]
)

PPO_model, train_PPO, val_PPO, gaps_PPO, losses_PPO = PPO_training(deepcopy(model), train_set_25, val_set_25;
    nb_epochs=200, batch_size=4, clip=0.2, sigma_values=[0.5, 0.5], lr_values = [1e-2, 1e-2]
)
# nb_epochs=400, batch_size=4, clip=0.2, sigma_values=[0.5, 0.1], lr_values = [1e-2, 1e-2]

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
IL_final_test_rew = [evaluate_solution(maximizer(IL_model(i.x); instance=i.instance), i.instance) for i in test_set_100]
JLD2.jldsave("logs/svsp_il_best_model.jld2"; model=IL_model, train_rew=train_IL, val_rew=val_IL, train_final=IL_final_train_rew, test_final=IL_final_test_rew)

PPO_final_train_rew = [evaluate_solution(maximizer(PPO_model(i.x); instance=i.instance), i.instance) for i in train_set_25]
PPO_final_test_rew = [evaluate_solution(maximizer(PPO_model(i.x); instance=i.instance), i.instance) for i in test_set_100]
JLD2.jldsave("logs/svsp_ppo_best_model.jld2"; model=PPO_model, train_rew=train_PPO, val_rew=val_PPO, train_final=PPO_final_train_rew, test_final=PPO_final_test_rew)