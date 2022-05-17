Random.seed!(67)

X, Y = generate_dataset(100)

dataset_path = tempname()
save_dataset(X, Y, dataset_path)
loaded_dataset = load_dataset(dataset_path)

@test 1. == 1.
