module StochasticVehicleScheduling

using Cbc
using ConstrainedShortestPaths
using Distributions
using Flux
using GLMakie
using GLPK
using Graphs
using InferOpt
using JLD2
using JuMP
using LinearAlgebra
using Logging
using MetaGraphs
using NamedTupleTools
using Printf
using ProgressMeter
using Random
using Requires
using SparseArrays
using Statistics
using Test
using TensorBoardLogger
using YAML

function __init__()
    @require Gurobi="2e9cd046-0924-5485-92f1-d5272153d98b" include("gurobi_stuff.jl")
end

include("utils/utils.jl")

include("instance_generation/default_values.jl")
include("instance_generation/task.jl")
include("instance_generation/district.jl")
include("instance_generation/city.jl")
include("instance_generation/instance.jl")

include("solution/solution.jl")
include("solution/deterministic_vsp.jl")
include("solution/local_search.jl")
include("solution/column_generation.jl")
include("solution/plne.jl")

include("dataset/dataset.jl")

include("visualization/visualization.jl")

include("training/dataset.jl")
include("training/trainer.jl")
include("training/metrics.jl")
include("training/perf.jl")

# Data strutures
export create_random_city
export Instance, Solution
export is_admissible

export evaluate_solution, evaluate_solution2
export Solution, solution_from_JuMP_array, basic_solution, get_routes

# Solvers
export cbc_model, glpk_model, grb_model
export solve_deterministic_VSP, easy_problem
export solve_scenario, solve_scenario2, solve_scenario3
export local_search, heuristic_solution
export column_generation, compute_solution_from_selected_columns
export solve_scenarios

# Dataset
export generate_dataset, save_dataset, load_dataset, normalize_data!, compute_μ_σ, reduce_data!

# Training
dropfirstdim(z::AbstractArray) = dropdims(z; dims=1)
make_negative(z::AbstractArray; threshold=0.) = -exp.(z) - threshold

export mape, normalized_mape
export hamming_distance, normalized_hamming_distance
export define_pipeline_loss
export plot_perf, test_perf
export dropfirstdim, make_negative
export train_test_split

export AbstractScalarMetric
export compute_metrics!
export Loss, HammingDistance, CostGap, ParameterError, MeanSquaredError

export read_config
export AbstractDataset, SupervisedDataset, ExperienceDataset
export Trainer, FenchelYoungGLM
export train_loop!

# Visualization
export plot_instance, plot_solution

export save_config

end
