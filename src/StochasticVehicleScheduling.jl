module StochasticVehicleScheduling

using Cbc
using ConstrainedShortestPaths
using Distributions
using GLMakie
using GLPK
using Graphs
using Gurobi
using JLD2
using JuMP
using MetaGraphs
using Printf
using ProgressMeter
using Random
using SparseArrays

# Gurobi package setup (see https://github.com/jump-dev/Gurobi.jl/issues/424)
const GRB_ENV = Ref{Gurobi.Env}()

function __init__()
    GRB_ENV[] = Gurobi.Env()
    return
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

include("training/Training.jl")

# Data strutures
export create_random_city
export Instance, Solution
export is_admissible

export evaluate_solution
export Solution, solution_from_JuMP_array, basic_solution

# Solvers
export cbc_model, glpk_model, grb_model
export solve_deterministic_VSP, easy_problem
export solve_scenario, solve_scenario2, solve_scenario3
export local_search, heuristic_solution
export column_generation, compute_solution_from_selected_columns
export solve_scenarios

# Dataset
export generate_dataset, save_dataset, load_dataset, normalize_data!

# Visualization
export plot_instance, plot_solution

end
