include("00_config.jl")

using Statistics

data = JLD2.load(dataset_path);
dataset_25 = data["dataset_25"];
dataset_50 = data["dataset_50"];
dataset_100 = data["dataset_100"];

res_data = JLD2.load(results_path)
supervised_model = res_data["supervised_model"]
experience_model = res_data["experience_model"]
supervised_loss_history = res_data["supervised_loss_history"]
experience_loss_history = res_data["experience_loss_history"]
supervised_gap_history = res_data["supervised_gap_history"]
experience_gap_history = res_data["experience_gap_history"]

# Evaluation
lineplot(supervised_loss_history)
lineplot(supervised_gap_history)
lineplot(experience_loss_history)
lineplot(experience_gap_history)

compute_gap(b, train_set_25, supervised_model, maximizer)
compute_gap(b, val_set_25, supervised_model, maximizer)
compute_gap(b, test_set_25, supervised_model, maximizer)
compute_gap(b, test_set_50, supervised_model, maximizer)
compute_gap(b, test_set_100, supervised_model, maximizer)

compute_gap(b, train_set_25, experience_model, maximizer)
compute_gap(b, val_set_25, experience_model, maximizer)
compute_gap(b, test_set_25, experience_model, maximizer)
compute_gap(b, test_set_50, experience_model, maximizer)
compute_gap(b, test_set_100, experience_model, maximizer)

op = maximum
compute_gap(b, train_set_25, supervised_model, maximizer, op)
compute_gap(b, val_set_25, supervised_model, maximizer, op)
compute_gap(b, test_set_25, supervised_model, maximizer, op)
compute_gap(b, test_set_50, supervised_model, maximizer, op)
compute_gap(b, test_set_100, supervised_model, maximizer, op)

# Cost comparison
opt_train = mean([evaluate_solution(i.y_true, i.instance) for i in train_set_25]) # 6884.853106548367
opt_test = mean([evaluate_solution(i.y_true, i.instance) for i in test_set_100]) # 20731.859918365783
SL_train = mean([evaluate_solution(maximizer(supervised_model(i.x); instance=i.instance), i.instance) for i in train_set_25]) # 6906.400775799109
SL_test = mean([evaluate_solution(maximizer(supervised_model(i.x); instance=i.instance), i.instance) for i in test_set_100]) # 21078.99508454414
RM_train = mean([evaluate_solution(maximizer(experience_model(i.x); instance=i.instance), i.instance) for i in train_set_25]) # 6881.930534591276
RM_test = mean([evaluate_solution(maximizer(experience_model(i.x); instance=i.instance), i.instance) for i in test_set_100]) # 20942.81209049026
SL_final_train_rew = [evaluate_solution(maximizer(supervised_model(i.x); instance=i.instance), i.instance) for i in train_set_25]
SL_final_test_rew = [evaluate_solution(maximizer(supervised_model(i.x); instance=i.instance), i.instance) for i in test_set_100]
JLD2.jldsave("logs/svsp_sl_best_model.jld2"; model=supervised_model, train_rew=supervised_train, val_rew=supervised_val, train_final=SL_final_train_rew, test_final=SL_final_test_rew)

RM_final_train_rew = [evaluate_solution(maximizer(experience_model(i.x); instance=i.instance), i.instance) for i in train_set_25]
RM_final_test_rew = [evaluate_solution(maximizer(experience_model(i.x); instance=i.instance), i.instance) for i in test_set_100]
opt_final_train_rew = [evaluate_solution(i.y_true, i.instance) for i in train_set_25]
opt_final_test_rew = [evaluate_solution(i.y_true, i.instance) for i in test_set_100]
JLD2.jldsave("logs/svsp_baselines.jld2"; greedy_train=RM_final_train_rew, optimal_train=opt_final_train_rew, greedy_test=RM_final_test_rew, optimal_test=opt_final_test_rew)