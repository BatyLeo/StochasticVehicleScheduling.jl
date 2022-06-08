module Training

using Flux
using InferOpt
using JLD2
using LinearAlgebra
using Logging
using NamedTupleTools
using ProgressMeter
using StochasticVehicleScheduling
using Statistics
using Test
using TensorBoardLogger
using YAML

include("config.jl")
include("dataset.jl")
#include("trainer.jl")
include("models.jl")
include("metrics.jl")
include("error.jl")

dropfirstdim(z::AbstractArray) = dropdims(z; dims=1)
make_negative(z::AbstractArray; threshold=0.) = -exp.(z) - threshold

export mape, normalized_mape
export hamming_distance, normalized_hamming_distance
export define_pipeline_loss
export plot_perf, test_perf
export dropfirstdim, make_negative
export train_test_split

#export InferOptTrainer, InferOptDataset
export AbstractScalarMetric
export compute_metrics!
export Loss, HammingDistance, CostGap, ParameterError, MeanSquaredError

export read_config
export AbstractDataset, SupervisedDataset, ExperienceDataset
export Trainer, FenchelYoungGLM
export train_loop!

end
