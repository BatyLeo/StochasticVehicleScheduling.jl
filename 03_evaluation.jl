include("00_config.jl")

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

op = maximum
compute_gap(b, train_set_25, supervised_model, maximizer, op)
compute_gap(b, val_set_25, supervised_model, maximizer, op)
compute_gap(b, test_set_25, supervised_model, maximizer, op)
compute_gap(b, test_set_50, supervised_model, maximizer, op)
compute_gap(b, test_set_100, supervised_model, maximizer, op)
