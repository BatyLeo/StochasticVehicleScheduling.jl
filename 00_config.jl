using DecisionFocusedLearningBenchmarks
using DecisionFocusedLearningBenchmarks.StochasticVehicleScheduling:
    evaluate_solution, compact_mip, local_search, deterministic_mip
using JLD2: JLD2
using StatsBase: StatsBase
using MLUtils: splitobs, numobs
using ProgressMeter: @showprogress
using UnicodePlots: lineplot

logdir = "logs"
mkpath(logdir)
dataset_path = joinpath(logdir, "datasets.jld2")
test_dataset_path = joinpath(logdir, "test_datasets.jld2")
results_path = joinpath(logdir, "results.jld2")
