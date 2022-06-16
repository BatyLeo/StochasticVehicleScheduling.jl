abstract type AbstractMetric end

function compute_value!(m::AbstractMetric, t::Trainer; kwargs...)
    push!(m.history, m(t; kwargs...))
end

function test_perf(metric::AbstractMetric)
    @test metric.history[end] < metric.history[1]
end

function log_last_measure!(m::AbstractMetric, logger=nothing; train=true, step_increment=0) end

## ---

struct Loss <: AbstractMetric
    name::String
    history::Vector{Float64}
end

Loss(name="Loss") = Loss(name, Float64[])

function (m::Loss)(trainer::Trainer; train, kwargs...)
    data = train ? trainer.data.train : trainer.data.test
    return mean(trainer.loss(t...) for t in loader(data))
end

function log_last_measure!(m::Loss, logger::AbstractLogger; train=true, step_increment=0)
    str = train ? "train" : "test"
    with_logger(logger) do
        @info "$str" loss=m.history[end] log_step_increment=step_increment
    end
end

## ---

# struct HammingDistance <: AbstractMetric
#     name::String
#     history::Vector{Float64}
# end

# HammingDistance(name="Hamming distance") = HammingDistance(name, Float64[])

# function (m::HammingDistance)(trainer::InferOptTrainer, data; Y_pred, kwargs...)
#     dist = mean(
#         hamming_distance(y, y_pred) for (y, y_pred) in zip(data.Y, Y_pred)
#     )
#     return dist
# end

# function test_perf(metric::HammingDistance)
#     @test metric.history[end] < metric.history[1] / 2
# end

# function log_last_measure!(m::HammingDistance, logger::AbstractLogger; train=true, step_increment=0)
#     str = train ? "train" : "test"
#     with_logger(logger) do
#         @info "$str" hamming=m.history[end] log_step_increment=step_increment
#     end
# end

## ---

struct AverageCostGap <: AbstractMetric
    name::String
    history::Vector{Float64}
end

AverageCostGap(name="Average cost gap") = AverageCostGap(name, Float64[])

function (m::AverageCostGap)(trainer::Trainer; train, Y_pred, kwargs...)
    (; cost) = trainer
    data = train ? trainer.data.train : trainer.data.test
    train_cost = [cost(y; instance=x) for (x, y) in zip(data.X, Y_pred)]
    train_cost_opt = [cost(y; instance=x) for (x, y) in zip(data.X, data.Y)]

    cost_gap = mean(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
    )
    return cost_gap
end

function log_last_measure!(m::AverageCostGap, logger::AbstractLogger; train=true, step_increment=0)
    str = train ? "train" : "test"
    with_logger(logger) do
        @info "$str" average_cost_gap=m.history[end] log_step_increment=step_increment
    end
end

## ---

struct MaxCostGap <: AbstractMetric
    name::String
    history::Vector{Float64}
end

MaxCostGap(name="Max cost gap") = MaxCostGap(name, Float64[])

function (m::MaxCostGap)(trainer::Trainer; train, Y_pred, kwargs...)
    (; cost) = trainer
    data = train ? trainer.data.train : trainer.data.test
    train_cost = [cost(y; instance=x) for (x, y) in zip(data.X, Y_pred)]
    train_cost_opt = [cost(y; instance=x) for (x, y) in zip(data.X, data.Y)]

    cost_gap = maximum(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
    )
    return cost_gap
end

function log_last_measure!(m::MaxCostGap, logger::AbstractLogger; train=true, step_increment=0)
    str = train ? "train" : "test"
    with_logger(logger) do
        @info "$str" max_cost_gap=m.history[end] log_step_increment=step_increment
    end
end
## ---

struct AverageCost <: AbstractMetric
    name::String
    history::Vector{Float64}
end

AverageCost(name="Average cost") = AverageCost(name, Float64[])

function (m::AverageCost)(trainer::Trainer; train, Y_pred, kwargs...)
    (; cost) = trainer
    data = train ? trainer.data.train : trainer.data.test
    train_cost = [cost(y; instance=x) for (x, y) in zip(data.X, Y_pred)]
    train_cost_opt = [cost(y; instance=x) for (x, y) in zip(data.X, data.Y)]

    cost_gap = mean(
        c for (c, c_opt) in zip(train_cost, train_cost_opt)
    )
    return cost_gap
end

function log_last_measure!(m::AverageCost, logger::AbstractLogger; train=true, step_increment=0)
    str = train ? "train" : "test"
    with_logger(logger) do
        @info "$str" average_cost=m.history[end] log_step_increment=step_increment
    end
end

##

struct ModelWeights{H} <: AbstractMetric
    name::String
    history::H
end

ModelWeights(name="Model") = ModelWeights(name, Dict{String, Any}[])

function fill_param_dict!(dict, m, prefix)
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix*"layer_"*string(i)*"/"*string(layer)*"/")
        end
    else
        for fieldname in fieldnames(typeof(m))
            val = getfield(m, fieldname)
            if val isa AbstractArray
                val = vec(val)
            end
            dict[prefix*string(fieldname)] = val
        end
    end
end

function (m::ModelWeights)(trainer::Trainer; kwargs...)
    param_dict = Dict{String, Any}()
    fill_param_dict!(param_dict, trainer.pipeline.encoder, "")
    return param_dict
end

function log_last_measure!(m::ModelWeights, logger::AbstractLogger; train=true, step_increment=0)
    with_logger(logger) do
        @info "model" params=m.history[end] log_step_increment=step_increment
    end
end

## ---

struct AveragePerturbedCostGap <: AbstractMetric
    name::String
    history::Vector{Float64}
end

AveragePerturbedCostGap(name="Average cost gap") = AveragePerturbedCostGap(name, Float64[])

function (m::AveragePerturbedCostGap)(trainer::Trainer; train, kwargs...)
    (; cost) = trainer
    data = train ? trainer.data.train : trainer.data.test

    perturbed = PerturbedNormal(trainer.pipeline.maximizer; ε=1000, M=5)

    thetas = [trainer.pipeline.encoder(x.features) for x in data.X]
    train_cost = zeros(length(thetas))
    for (i, (θ, x)) in enumerate(zip(thetas, data.X))
        mini = Inf
        for m in 1:10
            pert = InferOpt.sample_perturbation(perturbed, θ)
            mini = min(cost(trainer.pipeline.maximizer(pert; instance=x), instance=x), mini)
        end
        train_cost[i] = mini
    end

    #train_cost = [cost(y; instance=x) for (x, y) in zip(data.X, Y_pred)]
    train_cost_opt = [cost(y; instance=x) for (x, y) in zip(data.X, data.Y)]

    cost_gap = mean(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
    )
    return cost_gap
end

function log_last_measure!(m::AveragePerturbedCostGap, logger::AbstractLogger; train=true, step_increment=0)
    str = train ? "train" : "test"
    with_logger(logger) do
        @info "$str" average_perturbed_cost_gap=m.history[end] log_step_increment=step_increment
    end
end

# struct ParameterError <: AbstractMetric
#     name::String
#     history::Vector{Float64}
# end

# ParameterError(name="Parameter error") = ParameterError(name, Float64[])

# function (m::ParameterError)(trainer::InferOptTrainer, data; kwargs...)
#     (; true_encoder, encoder) = trainer.extra_info
#     w_true = first(true_encoder).weight
#     w_learned = first(encoder).weight
#     parameter_error = normalized_mape(w_true, w_learned)
#     return parameter_error
# end

# function test_perf(metric::ParameterError)
#     @test metric.history[end] < metric.history[1] / 2
# end

# function log_last_measure!(m::ParameterError, logger::AbstractLogger; train=true, step_increment=0)
#     str = train ? "train" : "test"
#     with_logger(logger) do
#         @info "$str" error=m.history[end] log_step_increment=step_increment
#     end
# end

# ## ---

# struct MeanSquaredError <: AbstractMetric
#     name::String
#     history::Vector{Float64}
# end

# MeanSquaredError(name="Mean squared error") = MeanSquaredError(name, Float64[])

# function (m::MeanSquaredError)(trainer::InferOptTrainer, data; Y_pred, kwargs...)
#     train_error = mean(
#         sum((y - y_pred) .^ 2) for (y, y_pred) in zip(data.Y, Y_pred)
#     )
#     return train_error
# end

# function test_perf(metric::MeanSquaredError)
#     @test metric.history[end] < metric.history[1] / 2
# end

# function log_last_measure!(m::MeanSquaredError, logger::AbstractLogger; train=true, step_increment=0)
#     str = train ? "train" : "test"
#     with_logger(logger) do
#         @info "$str" mse=m.history[end] log_step_increment=step_increment
#     end
# end

# ## ----

# struct ScalarMetric{R <: Real, F} <: AbstractMetric
#     name::String
#     history::Vector{R}
#     f::F
# end

# ScalarMetric(;name::String, f) = ScalarMetric(name, [], f)

# function (m::ScalarMetric)(t::InferOptTrainer)
#     return m.f(t)
# end

# function name(m::AbstractMetric)
#     return m.name
# end
