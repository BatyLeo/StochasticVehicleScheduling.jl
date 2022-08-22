# Reproduce paper experiments

All the scripts needed to reproduce experiments from the [paper](https://arxiv.org/abs/2207.13513) can be found in the [`scripts` folder](https://github.com/BatyLeo/StochasticVehicleScheduling.jl/tree/main/scripts).

Once downloaded, you need to instantiate the julia environment:

```bash
cd scripts
julia
```

Enter pkg mode by typing `]`
```
pkg> activate .
(scripts) pkg> instantiate
```

You you have Gurobi installed, you can also add the corresponding package and import it:
```
(scripts) pkg> add Gurobi
julia> using Gurobi
```

## 1. Generate datasets
The first script is `01_dataset.jl`, which generate all the datasets needed for the experiments. You can change the `settings` variable if you want to create different datasets.

## 2. Train models
The second script `02_training.jl` is the main script, that runs all the trainings. It uses the config files in the `configs` folder. If you want to run different experiments, you can either modify the config files, or create new ones and add them to the script. By default, training results will be written in the `logs` directory.

## 3. Visualize training plots
In order to visualize training plot evolution there are two options:
- Install tensorboard an run `tensorboard --logdir=logs`, which allow monitoring dynamically all the training plots during and after training.
- Run the `03_generate_plots.jl` script to generate static plots after the training.
## 4. Evaluate final models
The `04_evaluate_model.jl` script evaluates each of the models on all the test datasets. And write the results to `logs/results`

## 5. Generate result tables
Finally, `05_generate_tables.jl` reads from `logs/results` and creates readable tables in LaTeX format.
