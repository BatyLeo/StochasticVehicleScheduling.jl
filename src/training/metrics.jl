abstract type AbstractScalarMetric end

function compute_value!(m::AbstractScalarMetric, t::Trainer; kwargs...)
    push!(m.history, m(t; kwargs...))
end

function test_perf(metric::AbstractScalarMetric)
    @test metric.history[end] < metric.history[1]
end

function log_last_measure!(m::AbstractScalarMetric, logger=nothing; train=true, step_increment=0) end

## ---

struct Loss <: AbstractScalarMetric
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

# struct HammingDistance <: AbstractScalarMetric
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

struct AverageCostGap <: AbstractScalarMetric
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

struct MaxCostGap <: AbstractScalarMetric
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

struct AverageCost <: AbstractScalarMetric
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

## ---

# struct ParameterError <: AbstractScalarMetric
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

# struct MeanSquaredError <: AbstractScalarMetric
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

# struct ScalarMetric{R <: Real, F} <: AbstractScalarMetric
#     name::String
#     history::Vector{R}
#     f::F
# end

# ScalarMetric(;name::String, f) = ScalarMetric(name, [], f)

# function (m::ScalarMetric)(t::InferOptTrainer)
#     return m.f(t)
# end

# function name(m::AbstractScalarMetric)
#     return m.name
# end
