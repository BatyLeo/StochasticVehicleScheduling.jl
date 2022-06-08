using CUDA
using Flux
using Gurobi
using InferOpt
using JLD2
using JuMP
using Logging
using ProgressMeter
using Random
using StochasticVehicleScheduling
using StochasticVehicleScheduling.Training
using TensorBoardLogger
using UnicodePlots

const GRB_ENV = Gurobi.Env()

function grb_model()
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(model, "OutputFlag", 0)
    return model
end

function main()
    Random.seed!(67);

    config = read_config("src/training/config.yaml")
    trainer = FenchelYoungGLM(config; model_builder=grb_model)
    train_loop!(trainer, 10)

    return nothing
end

main()
