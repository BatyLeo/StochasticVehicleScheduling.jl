module StochasticVehicleScheduling

using Cbc
using Distributions
using GLPK
using Graphs
# using Gurobi
using JLD2
using JuMP
using MetaGraphs
using Random
using Printf
using ProgressMeter

# function __init__()
#     GRB_ENV[] = Gurobi.Env()
#     return
# end

# gurobi_optimizer() = Gurobi.Optimizer(GRB_ENV[])

# const GRB_ENV = Ref{Gurobi.Env}()

include("utils/utils.jl")

include("instance_generation/default_values.jl")
include("instance_generation/task.jl")
include("instance_generation/district.jl")
include("instance_generation/city.jl")
include("instance_generation/instance.jl")

include("solution/solution.jl")
include("solution/deterministic_vsp.jl")
include("solution/local_search.jl")

include("dataset/dataset.jl")

# Data strutures
export create_random_city
export Instance, Solution
export is_admissible

export evaluate_solution
export Solution, solution_from_JuMP_array, basic_solution

# Solvers
export solve_deterministic_VSP, easy_problem
export solve_scenario, solve_scenario2, solve_scenario3
export local_search, heuristic_solution

# Dataset
export generate_dataset, save_dataset, load_dataset

end
