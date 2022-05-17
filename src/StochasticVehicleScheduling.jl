module StochasticVehicleScheduling

using Distributions
using GLPK
using Graphs
using Gurobi
using JLD2
using JuMP
using MetaGraphs
using Random
using Printf
using ProgressMeter

function __init__()
    GRB_ENV[] = Gurobi.Env()
    return
end

gurobi_optimizer() = Gurobi.Optimizer(GRB_ENV[])

include("const.jl")
include("utils.jl")

include("task.jl")
include("district.jl")
include("city.jl")

include("instance.jl")
include("solution.jl")

include("deterministic_vsp.jl")
include("local_search.jl")

include("dataset.jl")

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
