module StochasticVehicleScheduling

using Cbc
using ConstrainedShortestPaths
using Distributions
using GLMakie
using GLPK
using Graphs
using JLD2
using JuMP
using MetaGraphs
using Printf
using ProgressMeter
using Random
using SparseArrays

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

# Data strutures
export create_random_city
export Instance, Solution
export is_admissible

export evaluate_solution
export Solution, solution_from_JuMP_array, basic_solution

# Solvers
export cbc_model, glpk_model
export solve_deterministic_VSP, easy_problem
export solve_scenario, solve_scenario2, solve_scenario3
export local_search, heuristic_solution
export column_generation, compute_solution_from_selected_columns
export solve_scenarios

# Dataset
export generate_dataset, save_dataset, load_dataset

# Visualization
export plot_instance, plot_solution

end
