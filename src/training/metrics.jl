# This file implements Metric type, which allow automatized metric computation

abstract type AbstractMetric end

function compute_value!(m::AbstractMetric, t::Trainer; kwargs...)
    return push!(m.history, m(t; kwargs...))
end

function test_perf(metric::AbstractMetric)
    @test metric.history[end] < metric.history[1]
end

function log_last_measure!(m::AbstractMetric, logger=nothing; train=true, step_increment=0) end

## ---

mutable struct Loss <: AbstractMetric
    name::String
    history::Vector{Float64}
    best_value::Float64
end

Loss(name="Loss") = Loss(name, Float64[], Inf)

function (m::Loss)(trainer::Trainer; train, epoch, kwargs...)
    data = train ? trainer.data.train : trainer.data.validation
    value = mean(trainer.loss(t...) for t in loader(data))
    return value
end

function log_last_measure!(m::Loss, logger::AbstractLogger; train=true, step_increment=0)
    str = train ? "train" : "validation"
    with_logger(logger) do
        @info "$str" loss = m.history[end] log_step_increment = step_increment
    end
end

## ---

mutable struct AllCosts <: AbstractMetric
    name::String
    best_value::Float64
    average_cost_gap::Vector{Float64}
    max_cost_gap::Vector{Float64}
    average_cost_per_task::Vector{Float64}
end

AllCosts(name="All costs") = AllCosts(name, Inf, Float64[], Float64[], Float64[])

function compute_value!(m::AllCosts, t::Trainer; kwargs...)
    a, b, c = m(t; kwargs...)
    push!(m.average_cost_gap, a)
    push!(m.max_cost_gap, b)
    return push!(m.average_cost_per_task, c)
end

function (m::AllCosts)(trainer::Trainer; train, epoch, Y_pred, kwargs...)
    (; cost) = trainer
    data = train ? trainer.data.train : trainer.data.validation
    train_cost = [cost(y; instance=x) for (x, y) in zip(data.X, Y_pred)]
    train_cost_opt = [cost(y; instance=x) for (x, y) in zip(data.X, data.Y)]

    average_cost_gap =
        mean((c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)) *
        100
    max_cost_gap =
        maximum(
            (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
        ) * 100
    average_cost_per_task = mean(
        c / x.city.nb_tasks for (c, c_opt, x) in zip(train_cost, train_cost_opt, data.X)
    )

    if !train && average_cost_per_task < m.best_value
        m.best_value = average_cost_per_task
        save_model(trainer, epoch; best=true)
    end
    return average_cost_gap, max_cost_gap, average_cost_per_task#, max_cost
end

function log_last_measure!(
    m::AllCosts, logger::AbstractLogger; train=true, step_increment=0
)
    str = train ? "train" : "validation"
    with_logger(logger) do
        @info "$str" average_cost_gap = m.average_cost_gap[end] log_step_increment =
            step_increment
        @info "$str" max_cost_gap = m.max_cost_gap[end] log_step_increment = 0
        @info "$str" average_cost_per_task = m.average_cost_per_task[end] log_step_increment =
            0
    end
end

## --

# struct AverageCostGap <: AbstractMetric
#     name::String
#     history::Vector{Float64}
# end

# AverageCostGap(name="Average cost gap") = AverageCostGap(name, Float64[])

# function (m::AverageCostGap)(trainer::Trainer; train, Y_pred, kwargs...)
#     (; cost) = trainer
#     data = train ? trainer.data.train : trainer.data.validation
#     train_cost = [cost(y; instance=x) for (x, y) in zip(data.X, Y_pred)]
#     train_cost_opt = [cost(y; instance=x) for (x, y) in zip(data.X, data.Y)]

#     cost_gap = mean(
#         (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
#     )
#     return cost_gap
# end

# function log_last_measure!(m::AverageCostGap, logger::AbstractLogger; train=true, step_increment=0)
#     str = train ? "train" : "validation"
#     with_logger(logger) do
#         @info "$str" average_cost_gap=m.history[end] log_step_increment=step_increment
#     end
# end

# ## ---

# struct MaxCostGap <: AbstractMetric
#     name::String
#     history::Vector{Float64}
# end

# MaxCostGap(name="Max cost gap") = MaxCostGap(name, Float64[])

# function (m::MaxCostGap)(trainer::Trainer; train, Y_pred, kwargs...)
#     (; cost) = trainer
#     data = train ? trainer.data.train : trainer.data.validation
#     train_cost = [cost(y; instance=x) for (x, y) in zip(data.X, Y_pred)]
#     train_cost_opt = [cost(y; instance=x) for (x, y) in zip(data.X, data.Y)]

#     cost_gap = maximum(
#         (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
#     )
#     return cost_gap
# end

# function log_last_measure!(m::MaxCostGap, logger::AbstractLogger; train=true, step_increment=0)
#     str = train ? "train" : "validation"
#     with_logger(logger) do
#         @info "$str" max_cost_gap=m.history[end] log_step_increment=step_increment
#     end
# end
# ## ---

# struct AverageCost <: AbstractMetric
#     name::String
#     history::Vector{Float64}
# end

# AverageCost(name="Average cost") = AverageCost(name, Float64[])

# function (m::AverageCost)(trainer::Trainer; train, Y_pred, kwargs...)
#     (; cost) = trainer
#     data = train ? trainer.data.train : trainer.data.validation
#     train_cost = [cost(y; instance=x) for (x, y) in zip(data.X, Y_pred)]
#     train_cost_opt = [cost(y; instance=x) for (x, y) in zip(data.X, data.Y)]

#     cost_gap = mean(
#         c for (c, c_opt) in zip(train_cost, train_cost_opt)
#     )

#     return cost_gap
# end

# function log_last_measure!(m::AverageCost, logger::AbstractLogger; train=true, step_increment=0)
#     str = train ? "train" : "validation"
#     with_logger(logger) do
#         @info "$str" average_cost=m.history[end] log_step_increment=step_increment
#     end
# end

##

# struct ModelWeights{H} <: AbstractMetric
#     name::String
#     history::H
# end

# ModelWeights(name="Model") = ModelWeights(name, Dict{String, Any}[])

# function fill_param_dict!(dict, m, prefix)
#     if m isa Chain
#         for (i, layer) in enumerate(m.layers)
#             fill_param_dict!(dict, layer, prefix*"layer_"*string(i)*"/"*string(layer)*"/")
#         end
#     else
#         for fieldname in fieldnames(typeof(m))
#             val = getfield(m, fieldname)
#             if val isa AbstractArray
#                 val = vec(val)
#             end
#             dict[prefix*string(fieldname)] = val
#         end
#     end
# end

# function (m::ModelWeights)(trainer::Trainer; kwargs...)
#     param_dict = Dict{String, Any}()
#     fill_param_dict!(param_dict, trainer.pipeline.encoder, "")
#     return param_dict
# end

# function log_last_measure!(m::ModelWeights, logger::AbstractLogger; train=true, step_increment=0)
#     with_logger(logger) do
#         @info "model" params=m.history[end] log_step_increment=step_increment
#     end
# end

# ## ---

# struct AveragePerturbedCostGap <: AbstractMetric
#     name::String
#     history::Vector{Float64}
# end

# AveragePerturbedCostGap(name="Average cost gap") = AveragePerturbedCostGap(name, Float64[])

# function (m::AveragePerturbedCostGap)(trainer::Trainer; train, kwargs...)
#     (; cost) = trainer
#     data = train ? trainer.data.train : trainer.data.validation

#     perturbed = PerturbedAdditive(trainer.pipeline.maximizer; ε=1000, nb_samples=5)

#     thetas = [trainer.pipeline.encoder(x.features) for x in data.X]
#     train_cost = zeros(length(thetas))
#     for (i, (θ, x)) in enumerate(zip(thetas, data.X))
#         mini = Inf
#         Zs = InferOpt.sample_perturbations(perturbed, θ)
#         for z in Zs
#             mini = min(cost(trainer.pipeline.maximizer(perturbed(θ, z; instance=x); instance=x), instance=x), mini)
#         end
#         train_cost[i] = mini
#     end

#     #train_cost = [cost(y; instance=x) for (x, y) in zip(data.X, Y_pred)]
#     train_cost_opt = [cost(y; instance=x) for (x, y) in zip(data.X, data.Y)]

#     cost_gap = mean(
#         (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
#     )
#     return cost_gap
# end

# function log_last_measure!(m::AveragePerturbedCostGap, logger::AbstractLogger; train=true, step_increment=0)
#     str = train ? "train" : "validation"
#     with_logger(logger) do
#         @info "$str" average_perturbed_cost_gap=m.history[end] log_step_increment=step_increment
#     end
# end
