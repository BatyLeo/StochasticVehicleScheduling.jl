module StochasticVehicleScheduling

using Cbc
using ConstrainedShortestPaths
using Distributions
using Flux
# using GLMakie
using GLPK
using Graphs
using HiGHS
using InferOpt
using JLD2
using JuMP
using LinearAlgebra
using Logging
using NamedTupleTools
using Printf
using ProgressMeter
using Random
using Requires
using SparseArrays
using Statistics
using TensorBoardLogger
using Test
using ValueHistories
using YAML

function __init__()
    @info "If you have Gurobi installed and want to use it, make sure to `using Gurobi` in order to enable it."
    @require Gurobi = "2e9cd046-0924-5485-92f1-d5272153d98b" include("gurobi_setup.jl")
end

include("utils/utils.jl")

include("instance/default_values.jl")
include("instance/task.jl")
include("instance/district.jl")
include("instance/city.jl")
include("instance/instance.jl")

include("solution/solution.jl")
include("solution/model_builders.jl")
include("solution/exact_algorithms/plne.jl")
include("solution/exact_algorithms/column_generation.jl")
include("solution/heuristic_algorithms/deterministic_vsp.jl")
include("solution/heuristic_algorithms/local_search.jl")

include("dataset/dataset.jl")

include("learning/easy_problem.jl")
include("learning/dataset.jl")
include("learning/trainer.jl")
include("learning/models.jl")
include("learning/metrics.jl")
include("learning/perf.jl")

# Data structures
export create_random_city, create_random_instance, create_random_compact_instance
export Instance, CompactInstance, Solution
export get_nb_scenarios, get_nb_tasks
export is_admissible

export evaluate_solution#, evaluate_solution2
export Solution, solution_from_JuMP_array, basic_solution, get_routes, solution_from_paths

# Solvers
export cbc_model, glpk_model, highs_model
export solve_deterministic_VSP, easy_problem
export local_search, heuristic_solution
export column_generation,
    compute_solution_from_selected_columns, column_generation_algorithm
export solve_scenarios

# Dataset
export generate_dataset, normalize_data!, compute_μ_σ, reduce_data!, generate_samples

# Training
export read_config
export save_config
export Trainer
export compute_metrics!
export train_loop!

end
